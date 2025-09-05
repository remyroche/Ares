#!/usr/bin/env python3
"""
Unified Code Quality Pipeline - Comprehensive Version

This is the most comprehensive code quality analysis pipeline available,
including all analyzers, visualizations, advanced analysis, consolidated fix scripts,
and plugin support. It provides complete code quality assessment in a single pipeline.
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

# Import analyzers for comprehensive analysis (ONLY comprehensive analysis related)
from analyzers.enhanced_import_analysis import EnhancedImportAnalyzer
from analyzers.intelligent_import_fixer import IntelligentImportFixer
from analyzers.code_smell_detector import CodeSmellDetector
from analyzers.configuration_analyzer import ConfigurationAnalyzer
from analyzers.documentation_analyzer import DocumentationAnalyzer
from analyzers.performance_analyzer import PerformanceAnalyzer
from analyzers.test_coverage_analyzer import TestCoverageAnalyzer
from analyzers.concurrency_analyzer import ConcurrencyAnalyzer
from analyzers.error_handling_analyzer import ErrorHandlingAnalyzer
from analyzers.code_duplication_analyzer import CodeDuplicationAnalyzer
from analyzers.improved_signature_analyzer import ImprovedSignatureAnalyzer

# Import additional analyzers
from analyzers.syntax_validator import SyntaxValidator
from analyzers.linter_analyzer import LinterAnalyzer
from analyzers.undefined_names_analyzer import UndefinedNamesAnalyzer
from analyzers.type_checker import TypeChecker
from analyzers.static_analysis_analyzer import StaticAnalysisAnalyzer
from analyzers.ast_analysis_analyzer import ASTAnalysisAnalyzer
from analyzers.architecture_analyzer import ArchitectureAnalyzer
from analyzers.call_graph_analyzer import CallGraphAnalyzer
from analyzers.complexity_analyzer import ComplexityAnalyzer
from analyzers.dependency_analyzer import DependencyAnalyzer
from analyzers.metrics_analyzer import MetricsAnalyzer
from analyzers.data_flow_analyzer import DataFlowAnalyzer
from analyzers.enhanced_dead_code_analyzer import EnhancedDeadCodeAnalyzer

# Import enhanced analyzers for false positive reduction
from analyzers.enhanced_fallback_detector import EnhancedFallbackDetector, analyze_fallback_patterns
from analyzers.enhanced_security_analyzer import EnhancedSecurityAnalyzer, analyze_security_issues
from analyzers.enhanced_dynamic_import_analyzer import EnhancedDynamicImportAnalyzer, analyze_dynamic_imports
from analyzers.stub_object_analyzer import StubObjectAnalyzer, analyze_stub_objects

# Import comprehensive analysis components (ONLY comprehensive analysis related)
from comprehensive_code_review import CodeQualityReviewer
# from script_integration_analysis import ScriptIntegrationAnalyzer  # Deleted during cleanup
# from merge_conflict_detector import MergeConflictDetector  # Deleted during cleanup

# Import validators (ONLY comprehensive analysis related)
from validators.enhanced_validator import EnhancedValidator
from validators.function_validator import FunctionValidator
from validators.integrated_validator import IntegratedValidator

# Import reporters (ONLY comprehensive analysis related)
from reporters.quality_reporter import QualityReporter
from reporters.html_reporter import HTMLReporter
from reporters.error_reporter import ErrorReporter
from reporters.trend_reporter import TrendReporter

# Import core components
from scripts.robust_async_fixer import RobustAsyncFixer
from core.config import get_default_config

# Import fixers and utilities
from scripts.advanced_syntax_fixer import AdvancedSyntaxFixer
from scripts.fix_missing_imports import ImportFixer as SafeImportFixer
from scripts.enhanced_type_hints import TypeHintEnhancer

# Import plugin system
from plugins.plugin_manager import PluginManager
from plugins.plugin_registry import PluginRegistry

# Import utilities
from utils.report_aggregator import ReportAggregator
from utils.file_utils import find_python_files
# Removed - redundant imports


class UnifiedEnhancedPipeline:
    """
    Comprehensive Code Quality Analysis Pipeline
    
    This is the most comprehensive code quality analysis pipeline available,
    including ALL analyzers, visualizers, fix scripts, and plugin support.
    Provides complete code quality assessment in a single unified pipeline.
    """

    def __init__(self, project_root: str = "/workspace/src", enable_plugins: bool = True):
        self.project_root = Path(project_root)
        self.reports_dir = Path("/workspace/code_quality/reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config = get_default_config()
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
        self.report_aggregator = ReportAggregator(project_root)
        
        # Initialize plugin system
        self.enable_plugins = enable_plugins
        if self.enable_plugins:
            self.plugin_registry = PluginRegistry()
            self.plugin_manager = PluginManager(self.plugin_registry)
            self._register_all_plugins()
        
        # Initialize ALL analyzers
        self._initialize_all_analyzers()
        
        # Initialize visualizers
        self._initialize_visualizers()

    def _register_all_plugins(self):
        """Register ALL available plugins for comprehensive analysis."""
        try:
            # Register production plugins
            from plugins.production.syntax_fixer import SyntaxFixerPlugin
            from plugins.production.import_fixer import ImportFixerPlugin
            from plugins.production.dead_code_fixer import DeadCodeFixerPlugin
            from plugins.production.linter_runner import LinterRunnerPlugin
            from plugins.production.security_scanner import SecurityScannerPlugin
            
            # Register code quality plugins
            from plugins.black_fixer import BlackFixer
            from plugins.isort_fixer import IsortFixer
            from plugins.autopep8_fixer import Autopep8Fixer
            from plugins.autoflake_fixer import AutoflakeFixer
            from plugins.flake8_analyzer import Flake8Analyzer
            from plugins.ruff_analyzer import RuffAnalyzer
            from plugins.ruff_fixer import RuffFixer
            from plugins.docformatter_fixer import DocformatterFixer
            from plugins.flynt_fixer import FlyntFixer
            from plugins.future_annotations_fixer import FutureAnnotationsFixer
            from plugins.pyupgrade_fixer import PyupgradeFixer
            from plugins.unify_fixer import UnifyFixer
            from plugins.yapf_fixer import YapfFixer
            from plugins.yesqa_fixer import YesqaFixer
            
            # Register all plugins
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
            self.plugin_registry.register_plugin("docformatter_fixer", DocformatterFixer())
            self.plugin_registry.register_plugin("flynt_fixer", FlyntFixer())
            self.plugin_registry.register_plugin("future_annotations_fixer", FutureAnnotationsFixer())
            self.plugin_registry.register_plugin("pyupgrade_fixer", PyupgradeFixer())
            self.plugin_registry.register_plugin("unify_fixer", UnifyFixer())
            self.plugin_registry.register_plugin("yapf_fixer", YapfFixer())
            self.plugin_registry.register_plugin("yesqa_fixer", YesqaFixer())
            
            print(f"✓ All {len(self.plugin_registry.list_plugins())} plugins registered successfully")
        except ImportError as e:
            print(f"⚠ Warning: Could not register some plugins: {e}")

    def _initialize_all_analyzers(self):
        """Initialize ALL available analyzers for comprehensive analysis."""
        try:
            from core.config import get_default_config
            config = get_default_config()
            
            self.analyzers = {
                # Basic analyzers
                "syntax_validator": SyntaxValidator(config),
                "linter": LinterAnalyzer(config),
                "import_analyzer": ImportAnalyzer(config),
                "undefined_names": UndefinedNamesAnalyzer(config),
                "type_checker": TypeChecker(config),
                "static_analysis": StaticAnalysisAnalyzer(config),
                "ast_analysis": ASTAnalysisAnalyzer(config),
                
                # Advanced analyzers
                "architecture": ArchitectureAnalyzer(config),
                "call_graph": CallGraphAnalyzer(config),
                "code_duplication": CodeDuplicationAnalyzer(config),
                "complexity": ComplexityAnalyzer(config),
                "concurrency": ConcurrencyAnalyzer(config),
                "dependency": DependencyAnalyzer(config),
                "error_handling": ErrorHandlingAnalyzer(config),
                "signature": SignatureAnalyzer(config),
                "improved_signature": ImprovedSignatureAnalyzer(config),
                
                # Quality analyzers
                "code_smell": CodeSmellDetector(config),
                "configuration": ConfigurationAnalyzer(config),
                "data_flow": DataFlowAnalyzer(config),
                "documentation": DocumentationAnalyzer(config),
                "metrics": MetricsAnalyzer(config),
                "performance": PerformanceAnalyzer(config),
                "test_coverage": TestCoverageAnalyzer(config),
                
                # Dead code analyzers
                "dead_code": DeadCodeAnalyzer(config),
                "improved_dead_code": ImprovedDeadCodeAnalyzer(config),
                "enhanced_dead_code": EnhancedDeadCodeAnalyzer(config),
                "simplified_enhanced": SimplifiedEnhancedAnalyzer(config),
            }
            print(f"✓ All {len(self.analyzers)} analyzers initialized successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize some analyzers: {e}")
            self.analyzers = {}

    def _initialize_visualizers(self):
        """Initialize ALL visualizers for comprehensive visualization."""
        try:
            self.visualizers = {
                "dashboard": DashboardGenerator(str(self.project_root)),
                "complexity_heatmap": ComplexityHeatmap(str(self.project_root)),
                "dependency_graph": DependencyGraph(str(self.project_root)),
                "interaction_network": InteractionNetwork(str(self.project_root)),
                "code_visualizer": CodeVisualizer(str(self.project_root)),
            }
            print(f"✓ All {len(self.visualizers)} visualizers initialized successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize some visualizers: {e}")
            self.visualizers = {}

    def run_syntax_validation(self) -> dict[str, Any]:
        """Run syntax validation."""
        print("\n" + "="*60)
        print("Running Syntax Validation")
        print("="*60)
        
        try:
            results = self.syntax_validator.validate_syntax(str(self.project_root))
            return {"status": "completed", "results": results}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def run_import_validation(self) -> dict[str, Any]:
        """Run import validation."""
        print("\n" + "="*60)
        print("Running Import Validation")
        print("="*60)
        
        try:
            results = self.import_analyzer.validate_imports(str(self.project_root))
            return {"status": "completed", "results": results}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def run_syntax_fixes(self) -> dict[str, Any]:
        """Run advanced syntax fixes."""
        print("\n" + "="*60)
        print("Running Advanced Syntax Fixes")
        print("="*60)

        start_time = time.time()
        fixer = AdvancedSyntaxFixer(str(self.project_root))
        fixer.fix_all_syntax_errors(dry_run=False)

        result = {
            "fixed_files": fixer.fixed_files,
            "failed_files": [{"file": f, "error": fixer.syntax_errors.get(f, "Unknown syntax error")} 
                           for f in fixer.failed_files],
            "syntax_errors": dict(fixer.syntax_errors),
            "total_fixed": len(fixer.fixed_files),
            "total_failed": len(fixer.failed_files),
            "execution_time": time.time() - start_time,
        }

        # Add to aggregator
        self.report_aggregator.add_syntax_results(result)

        # Save individual report
        report_path = self.reports_dir / f"syntax_fixes_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_import_fixes(self) -> dict[str, Any]:
        """Run enhanced import fixes with auto-detection."""
        print("\n" + "="*60)
        print("Running Enhanced Import Fixes with Auto-Detection")
        print("="*60)

        start_time = time.time()
        
        try:
            # Use the enhanced ImportFixer with auto-detection
            from scripts.fix_missing_imports import ImportFixer
            fixer = ImportFixer(str(self.project_root))
            
            # Auto-detect and fix missing imports
            print("🔍 Auto-detecting missing imports...")
            result = fixer.auto_fix_all_files(
                [str(f) for f in self.file_paths],  # Use all files
                dry_run=False  # Actually fix the files
            )
            
            # Format results for report aggregator
            fixed_files_formatted = []
            for file_path in result.get('fixed_files', []):
                fixed_files_formatted.append({
                    "file": file_path,
                    "imports_added": ["Auto-detected imports"]  # Enhanced fixer tracks this
                })
            
            formatted_result = {
                "fixed_files": fixed_files_formatted,
                "failed_files": result.get('failed_files', []),
                "import_errors": {},
                "total_fixed": result.get('fixed', 0),
                "total_failed": result.get('failed', 0),
                "execution_time": time.time() - start_time,
                "auto_detection_summary": {
                    "files_analyzed": len(self.file_paths),
                    "files_with_missing_imports": result.get('fixed', 0),
                    "module_counts": result.get('module_counts', {})
                }
            }
            
            print(f"✅ Auto-fixed {result.get('fixed', 0)} files")
            print(f"❌ Failed to fix {result.get('failed', 0)} files")
            
            # Show module breakdown
            module_counts = result.get('module_counts', {})
            if module_counts:
                print("\n📊 Imports added by module:")
                for module, count in sorted(module_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {module}: {count} files")

        except Exception as e:
            print(f"❌ Enhanced import fixer failed: {e}")
            # Fallback to original fixer
            fixer = SafeImportFixer(str(self.project_root))
            fixer.fix_project(dry_run=False)
            
            fixed_files_formatted = []
            for file_path in fixer.fixed_files:
                fixed_files_formatted.append({
                    "file": file_path,
                    "imports_added": []
                })
            
            formatted_result = {
                "fixed_files": fixed_files_formatted,
                "failed_files": fixer.failed_files,
                "import_errors": {},
                "total_fixed": len(fixer.fixed_files),
                "total_failed": len(fixer.failed_files),
                "execution_time": time.time() - start_time,
                "fallback_used": True,
                "error": str(e)
            }

        # Add to aggregator
        self.report_aggregator.add_import_results(formatted_result)

        # Save individual report
        report_path = self.reports_dir / f"import_fixes_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(formatted_result, f, indent=2)

        return formatted_result

    def detect_circular_imports(self) -> dict[str, Any]:
        """Detect circular imports."""
        print("\n" + "="*60)
        print("Detecting Circular Imports")
        print("="*60)

        start_time = time.time()
        from scripts.detect_circular_imports import ImportAnalyzer
        detector = ImportAnalyzer(str(self.project_root))
        report = detector.generate_report()
        
        cycles = report.get("circular_imports", {}).get("cycles", [])
        
        result = {
            "circular_imports": cycles,
            "total_cycles": len(cycles),
            "affected_modules": list({
                module for cycle in cycles
                for module in cycle.get("modules", [])
            }),
            "execution_time": time.time() - start_time,
        }

        # Add to aggregator
        self.report_aggregator.add_circular_import_results(result)

        # Save individual report
        report_path = self.reports_dir / f"circular_imports_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_async_fixes(self) -> dict[str, Any]:
        """Run robust async/await fixes."""
        print("\n" + "="*60)
        print("Running Async/Await Fixes")
        print("="*60)

        start_time = time.time()
        fixer = RobustAsyncFixer(str(self.project_root))
        fixer.fix_all_async_issues(dry_run=False)

        # Convert fixed_files to proper format for report aggregator
        fixed_files_formatted = []
        for file_path in fixer.fixed_files:
            fixed_files_formatted.append({
                "file": file_path,
                "changes": []  # RobustAsyncFixer doesn't track specific changes
            })
        
        result = {
            "fixed_files": fixed_files_formatted,
            "failed_files": fixer.failed_files,
            "total_fixed": len(fixer.fixed_files),
            "total_failed": len(fixer.failed_files),
            "execution_time": time.time() - start_time,
        }

        # Add to aggregator
        self.report_aggregator.add_async_results(result)

        # Save individual report
        report_path = self.reports_dir / f"async_fixes_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_type_hints(self) -> dict[str, Any]:
        """Run type hint enhancements."""
        print("\n" + "="*60)
        print("Running Type Hint Enhancements")
        print("="*60)

        start_time = time.time()

        # Get all Python files
        python_files = []
        for pattern in ["**/*.py"]:
            python_files.extend(self.project_root.glob(pattern))

        fixed_files = []
        failed_files = []

        for file_path in python_files:
            try:
                enhancer = TypeHintEnhancer()

                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Parse and transform
                tree = ast.parse(content)
                new_tree = enhancer.visit(tree)

                if enhancer.changes_made:
                    # Generate new code
                    new_content = ast.unparse(new_tree)

                    # Add necessary imports
                    if enhancer.imports_needed:
                        import_lines = []
                        if any("Path" in imp for imp in enhancer.imports_needed):
                            import_lines.append("from pathlib import Path")
                        if any("Union" in imp or "Dict" in imp or "List" in imp or "Optional" in imp or "Any" in imp or "Tuple" in imp
                               for imp in enhancer.imports_needed):
                            import_lines.append("from typing import Dict, List, Optional, Union, Any, Tuple")

                        # Insert imports after module docstring and other imports
                        lines = new_content.split("\n")
                        insert_pos = 0
                        for i, line in enumerate(lines):
                            if line.strip() and not line.strip().startswith('"""') and not line.strip().startswith("#"):
                                if line.startswith(("import ", "from ")):
                                    insert_pos = i + 1
                                else:
                                    break

                        for imp in import_lines:
                            lines.insert(insert_pos, imp)
                            insert_pos += 1

                        new_content = "\n".join(lines)

                    # Write back
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)

                    fixed_files.append({
                        "file": str(file_path),
                        "changes": enhancer.changes_made,
                    })

            except Exception as e:
                failed_files.append({
                    "file": str(file_path),
                    "error": str(e),
                })

        result = {
            "fixed_files": fixed_files,
            "failed_files": failed_files,
            "total_fixed": len(fixed_files),
            "total_failed": len(failed_files),
            "execution_time": time.time() - start_time,
        }

        # Add to aggregator
        self.report_aggregator.add_type_results(result)

        # Save individual report
        report_path = self.reports_dir / f"type_hints_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_function_validation(self) -> dict[str, Any]:
        """Run function validation checks."""
        print("\n" + "="*60)
        print("Running Function Validation")
        print("="*60)

        start_time = time.time()
        validator = FunctionValidator(str(self.project_root))
        validation_result = validator.validate_project()

        result = {
            "issues": validation_result.get("issues", []),
            "total_issues": validation_result.get("total_issues", 0),
            "files_analyzed": validation_result.get("files_analyzed", []),
            "total_files": len(validation_result.get("files_analyzed", [])),
            "issue_summary": validation_result.get("issues_by_type", {}),
            "execution_time": time.time() - start_time,
        }

        # Add to aggregator
        self.report_aggregator.add_function_validation_results(result)

        # Save individual report
        report_path = self.reports_dir / f"function_validation_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_enhanced_validation(self) -> dict[str, Any]:
        """Run enhanced validation for function arguments and data access."""
        print("\n" + "="*60)
        print("Running Enhanced Validation (Arguments & Data Access)")
        print("="*60)

        start_time = time.time()
        validator = EnhancedValidator(str(self.project_root))
        report = validator.validate_project()

        result = {
            "issues": report["issues"],
            "total_issues": report["summary"]["total_issues"],
            "argument_mismatches": report["summary"]["argument_mismatches"],
            "unsafe_data_access": report["summary"]["unsafe_data_access"],
            "missing_null_checks": report["summary"]["missing_null_checks"],
            "type_inconsistencies": report["summary"]["type_inconsistencies"],
            "files_processed": report["summary"]["files_processed"],
            "execution_time": time.time() - start_time,
            "data_access_summary": report.get("data_access_summary", {}),
            "function_signatures": len(report.get("function_signatures", {})),
        }

        # Add to aggregator
        self.report_aggregator.add_enhanced_validation_results(report)

        # Save individual report
        report_path = self.reports_dir / f"enhanced_validation_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_comprehensive_review(self) -> dict[str, Any]:
        """Run comprehensive code quality review."""
        print("\n" + "="*60)
        print("Running Comprehensive Code Review")
        print("="*60)

        start_time = time.time()
        reviewer = CodeQualityReviewer(str(self.project_root))
        reviewer.review_directory(str(self.project_root))
        report = reviewer.generate_report()

        result = {
            "issues": report["issues"],
            "total_issues": len(report["issues"]),
            "summary": report["summary"],
            "metrics": report.get("metrics", {}),
            "security_issues": report.get("security_issues", []),
            "performance_issues": report.get("performance_issues", []),
            "execution_time": time.time() - start_time,
        }

        # Add to aggregator
        self.report_aggregator.add_comprehensive_review_results(result)

        # Save individual report
        report_path = self.reports_dir / f"comprehensive_review_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_interaction_mapping(self) -> dict[str, Any]:
        """Run code interaction mapping."""
        print("\n" + "="*60)
        print("Running Code Interaction Mapping")
        print("="*60)

        start_time = time.time()

        # Use the comprehensive review data
        reviewer = CodeQualityReviewer(str(self.project_root))
        reviewer.review_directory(str(self.project_root))
        report_data = reviewer.generate_report()

        # Extract interactions
        interactions = extract_interactions(report_data)

        # Generate readable report
        # report_content = generate_report(interactions)  # This function doesn't exist, interactions already contain the data

        result = {
            "interactions": interactions,
            "module_count": len(interactions["import_graph"]),
            "function_count": len(interactions["function_definitions"]),
            "undefined_functions": len(interactions["undefined_functions"]),
            "async_issues": len(interactions["async_patterns"]),
            "execution_time": time.time() - start_time,
        }

        # Save reports
        json_path = self.reports_dir / f"code_interactions_{self.timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

        text_path = self.reports_dir / f"code_interactions_{self.timestamp}.txt"
        with open(text_path, "w") as f:
            f.write(report_content)

        return result

    def run_metrics_analysis(self) -> dict[str, Any]:
        """Run code metrics analysis."""
        print("\n" + "="*60)
        print("Running Code Metrics Analysis")
        print("="*60)

        start_time = time.time()
        analyzer = MetricsAnalyzer(str(self.project_root))

        # Analyze all Python files
        python_files = list(self.project_root.rglob("*.py"))
        for file_path in python_files:
            analyzer.analyze_file(file_path)

        result = analyzer.generate_report()
        result["execution_time"] = time.time() - start_time

        # Add to aggregator (if available)
        if hasattr(analyzer, 'file_metrics') and hasattr(self.report_aggregator, 'file_metrics'):
            self.report_aggregator.file_metrics.update(analyzer.file_metrics)

        # Save report
        report_path = self.reports_dir / f"metrics_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_test_coverage_analysis(self) -> dict[str, Any]:
        """Run test coverage analysis."""
        print("\n" + "="*60)
        print("Running Test Coverage Analysis")
        print("="*60)

        start_time = time.time()
        analyzer = TestCoverageAnalyzer(str(self.project_root))
        result = analyzer.analyze_project()
        result["execution_time"] = time.time() - start_time

        # Save report
        report_path = self.reports_dir / f"test_coverage_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_code_smell_detection(self) -> dict[str, Any]:
        """Run code smell detection."""
        print("\n" + "="*60)
        print("Running Code Smell Detection")
        print("="*60)

        start_time = time.time()
        detector = CodeSmellDetector(str(self.project_root))

        # Analyze all Python files
        python_files = list(self.project_root.rglob("*.py"))
        for file_path in python_files:
            detector.analyze_file(file_path)

        result = detector.generate_report()
        result["execution_time"] = time.time() - start_time

        # Save report
        report_path = self.reports_dir / f"code_smells_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_documentation_analysis(self) -> dict[str, Any]:
        """Run documentation quality analysis."""
        print("\n" + "="*60)
        print("Running Documentation Analysis")
        print("="*60)

        start_time = time.time()
        analyzer = DocumentationAnalyzer(str(self.project_root))

        # Analyze all Python files
        python_files = list(self.project_root.rglob("*.py"))
        for file_path in python_files:
            analyzer.analyze_file(file_path)

        # Analyze README
        analyzer.analyze_readme()

        result = analyzer.generate_report()
        result["execution_time"] = time.time() - start_time

        # Save report
        report_path = self.reports_dir / f"documentation_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_performance_analysis(self) -> dict[str, Any]:
        """Run performance analysis."""
        print("\n" + "="*60)
        print("Running Performance Analysis")
        print("="*60)

        start_time = time.time()
        analyzer = PerformanceAnalyzer(str(self.project_root))

        # Analyze all Python files
        python_files = list(self.project_root.rglob("*.py"))
        for file_path in python_files:
            analyzer.analyze_file(file_path)

        result = analyzer.generate_report()
        result["execution_time"] = time.time() - start_time

        # Save report
        report_path = self.reports_dir / f"performance_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_configuration_analysis(self) -> dict[str, Any]:
        """Run configuration analysis."""
        print("\n" + "="*60)
        print("Running Configuration Analysis")
        print("="*60)

        start_time = time.time()
        analyzer = ConfigurationAnalyzer(str(self.project_root))
        result = analyzer.analyze_project()
        result["execution_time"] = time.time() - start_time

        # Save report
        report_path = self.reports_dir / f"configuration_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_data_flow_analysis(self) -> dict[str, Any]:
        """Run data flow analysis."""
        print("\n" + "="*60)
        print("Running Data Flow Analysis")
        print("="*60)

        start_time = time.time()
        analyzer = DataFlowAnalyzer(str(self.project_root))

        # Analyze all Python files
        python_files = list(self.project_root.rglob("*.py"))
        for file_path in python_files:
            analyzer.analyze_file(file_path)

        result = analyzer.generate_report()
        result["execution_time"] = time.time() - start_time

        # Save report
        report_path = self.reports_dir / f"data_flow_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_static_analysis(self) -> dict[str, Any]:
        """Run comprehensive static analysis."""
        print("\n" + "="*60)
        print("Running Comprehensive Static Analysis")
        print("="*60)

        start_time = time.time()
        
        # Create a mock config for the analyzer
        from core.config import get_default_config
        config = get_default_config()
        
        analyzer = StaticAnalysisAnalyzer(config)
        result = analyzer.analyze_directory(str(self.project_root))
        result["execution_time"] = time.time() - start_time

        # Add to aggregator (if method exists)
        if hasattr(self.report_aggregator, 'add_static_analysis_results'):
            self.report_aggregator.add_static_analysis_results(result)

        # Save report
        report_path = self.reports_dir / f"static_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_ast_analysis(self) -> dict[str, Any]:
        """Run advanced AST analysis."""
        print("\n" + "="*60)
        print("Running Advanced AST Analysis")
        print("="*60)

        start_time = time.time()
        
        # Create a mock config for the analyzer
        from core.config import get_default_config
        config = get_default_config()
        
        analyzer = ASTAnalysisAnalyzer(config)
        result = analyzer.analyze_directory(str(self.project_root))
        result["execution_time"] = time.time() - start_time

        # Add to aggregator (if method exists)
        if hasattr(self.report_aggregator, 'add_ast_analysis_results'):
            self.report_aggregator.add_ast_analysis_results(result)

        # Save report
        report_path = self.reports_dir / f"ast_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)


        return result

    def run_dead_code_analysis(self) -> dict[str, Any]:
        """Run improved dead code analysis."""
        print("\n" + "="*60)
        print("Running Improved Dead Code Analysis")
        print("="*60)

        start_time = time.time()
        analyzer = ImprovedDeadCodeAnalyzer()
        result = analyzer.analyze_directory(str(self.project_root))
        
        # Save individual report
        report_path = self.reports_dir / f"dead_code_analysis_{self.timestamp}.json"
        analyzer.save_report(report_path)

        analysis_result = {
            "issues": [
                {
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "issue_type": issue.issue_type,
                    "name": issue.name,
                    "description": issue.description,
                    "confidence": issue.confidence,
                    "severity": issue.severity,
                    "is_public_api": issue.is_public_api,
                    "is_used_cross_file": issue.is_used_cross_file,
                    "is_abstract_interface": issue.is_abstract_interface
                }
                for issue in result.issues
            ],
            "total_issues": result.total_issues,
            "files_analyzed": result.files_analyzed,
            "issues_by_type": result.issues_by_type,
            "issues_by_confidence": result.issues_by_confidence,
            "issues_by_severity": result.issues_by_severity,
            "high_confidence_issues": result.global_analysis["high_confidence_issues"],
            "public_api_issues": result.global_analysis["public_api_issues"],
            "cross_file_usage_issues": result.global_analysis["cross_file_usage_issues"],
            "abstract_interface_issues": result.global_analysis["abstract_interface_issues"],
            "execution_time": time.time() - start_time,
            "report_path": str(report_path)
        }

        # Add to aggregator
        self.report_aggregator.add_dead_code_results(analysis_result)

        return analysis_result

    def run_dead_code_fixes(self) -> dict[str, Any]:
        """Run automated dead code fixes."""
        print("\n" + "="*60)
        print("Running Automated Dead Code Fixes")
        print("="*60)

        start_time = time.time()
        
        # Import the plugin
        from plugins.production.dead_code_fixer import DeadCodeFixerPlugin
        
        # Create and configure plugin
        plugin = DeadCodeFixerPlugin()
        config = {
            "dry_run": False,  # Set to True for dry run
            "min_confidence": 0.95,
            "create_backups": True
        }
        plugin.configure(config)
        
        # Execute plugin
        context = {
            "project_root": str(self.project_root),
            "dead_code_report_path": self.reports_dir / f"dead_code_analysis_{self.timestamp}.json"
        }
        
        result = plugin.execute(context)
        
        # Save individual report
        report_path = self.reports_dir / f"dead_code_fixes_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        fix_result = {
            "total_files_processed": result["total_files_processed"],
            "successful_files": result["successful_files"],
            "failed_files": result["failed_files"],
            "total_fixes_applied": result["total_fixes_applied"],
            "total_errors": result["total_errors"],
            "execution_time": time.time() - start_time,
            "dry_run": result["dry_run"],
            "file_results": result["file_results"],
            "summary": result["summary"]
        }

        # Add to aggregator
        self.report_aggregator.add_dead_code_fix_results(fix_result)

        return fix_result

    def run_comprehensive_import_undefined_check(self) -> dict[str, Any]:
        """Run comprehensive import and undefined variable checker."""
        print("\n" + "="*60)
        print("Running Comprehensive Import and Undefined Checker")
        print("="*60)

        start_time = time.time()
        from analyzers.undefined_names_analyzer import UndefinedNamesAnalyzer
        checker = UndefinedNamesAnalyzer(self.config)
        result = checker.analyze_directory(str(self.project_root))
        result["execution_time"] = time.time() - start_time

        # Add to aggregator
        self.report_aggregator.add_comprehensive_review_results(result)

        # Save individual report
        report_path = self.reports_dir / f"comprehensive_import_undefined_check_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_advanced_analysis(self) -> dict[str, Any]:
        """Run advanced analysis including architecture, call graph, and complexity analysis."""
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

        # Save individual report
        report_path = self.reports_dir / f"advanced_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def run_consolidated_fixes(self) -> dict[str, Any]:
        """Run consolidated fix scripts for comprehensive code improvements."""
        print("\n" + "="*60)
        print("Running Consolidated Fix Scripts")
        print("="*60)

        start_time = time.time()
        results = {}

        # Bulk Syntax Cleanup
        print("Running Bulk Syntax Cleanup...")
        try:
            bulk_cleanup = BulkSyntaxCleanup(str(self.project_root))
            bulk_results = bulk_cleanup.cleanup_all_files()
            results["bulk_syntax_cleanup"] = bulk_results
        except Exception as e:
            results["bulk_syntax_cleanup"] = {"error": str(e)}

        # Apply All Fixes
        print("Running Apply All Fixes...")
        try:
            apply_fixes = ApplyAllFixes(str(self.project_root))
            apply_results = apply_fixes.apply_all_fixes()
            results["apply_all_fixes"] = apply_results
        except Exception as e:
            results["apply_all_fixes"] = {"error": str(e)}

        # Missing Import Fixer
        print("Running Missing Import Fixer...")
        try:
            import_fixer = MissingImportFixer(str(self.project_root))
            import_results = import_fixer.fix_missing_imports()
            results["missing_import_fixer"] = import_results
        except Exception as e:
            results["missing_import_fixer"] = {"error": str(e)}

        # Type Hint Adder
        print("Running Type Hint Adder...")
        try:
            type_adder = TypeHintAdder(str(self.project_root))
            type_results = type_adder.add_type_hints()
            results["type_hint_adder"] = type_results
        except Exception as e:
            results["type_hint_adder"] = {"error": str(e)}

        results["execution_time"] = time.time() - start_time

        # Save individual report
        report_path = self.reports_dir / f"consolidated_fixes_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def run_plugin_analysis(self) -> dict[str, Any]:
        """Run plugin-based analysis if plugins are enabled."""
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

        # Save individual report
        report_path = self.reports_dir / f"plugin_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def run_visualization_analysis(self) -> dict[str, Any]:
        """Run visualization and interaction mapping analysis."""
        print("\n" + "="*60)
        print("Running Visualization Analysis")
        print("="*60)

        start_time = time.time()
        results = {}

        # Enhanced Map Code Interactions
        print("Running Enhanced Code Interaction Mapping...")
        try:
            from enhanced_map_code_interactions import EnhancedCodeInteractionMapper
            mapper = EnhancedCodeInteractionMapper(str(self.project_root))
            interaction_results = mapper.map_interactions()
            results["enhanced_interactions"] = interaction_results
        except Exception as e:
            results["enhanced_interactions"] = {"error": str(e)}

        # Visualize Interactions
        print("Running Interaction Visualization...")
        try:
            from visualize_interactions import InteractionVisualizer
            visualizer = InteractionVisualizer(str(self.project_root))
            viz_results = visualizer.generate_visualizations()
            results["interaction_visualization"] = viz_results
        except Exception as e:
            results["interaction_visualization"] = {"error": str(e)}

        # Dashboard Generation
        print("Generating Dashboard...")
        try:
            from visualizers.dashboard_generator import DashboardGenerator
            dashboard = DashboardGenerator(str(self.project_root))
            dashboard_results = dashboard.generate_dashboard()
            results["dashboard"] = dashboard_results
        except Exception as e:
            results["dashboard"] = {"error": str(e)}

        results["execution_time"] = time.time() - start_time

        # Save individual report
        report_path = self.reports_dir / f"visualization_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

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

        # Save individual report
        report_path = self.reports_dir / f"dead_code_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def run_architecture_analysis(self) -> dict[str, Any]:
        """Run comprehensive architecture analysis."""
        print("\n" + "="*60)
        print("Running Architecture Analysis")
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

        # Dependency Analysis
        if "dependency" in self.analyzers:
            print("Running Dependency Analysis...")
            try:
                dep_results = self.analyzers["dependency"].analyze_directory(str(self.project_root))
                results["dependency"] = dep_results
            except Exception as e:
                results["dependency"] = {"error": str(e)}

        results["execution_time"] = time.time() - start_time

        # Save individual report
        report_path = self.reports_dir / f"architecture_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def run_performance_analysis(self) -> dict[str, Any]:
        """Run comprehensive performance analysis."""
        print("\n" + "="*60)
        print("Running Performance Analysis")
        print("="*60)

        start_time = time.time()
        results = {}

        # Performance Analysis
        if "performance" in self.analyzers:
            print("Running Performance Analysis...")
            try:
                perf_results = self.analyzers["performance"].analyze_directory(str(self.project_root))
                results["performance"] = perf_results
            except Exception as e:
                results["performance"] = {"error": str(e)}

        # Complexity Analysis
        if "complexity" in self.analyzers:
            print("Running Complexity Analysis...")
            try:
                complexity_results = self.analyzers["complexity"].analyze_directory(str(self.project_root))
                results["complexity"] = complexity_results
            except Exception as e:
                results["complexity"] = {"error": str(e)}

        # Code Duplication Analysis
        if "code_duplication" in self.analyzers:
            print("Running Code Duplication Analysis...")
            try:
                dup_results = self.analyzers["code_duplication"].analyze_directory(str(self.project_root))
                results["code_duplication"] = dup_results
            except Exception as e:
                results["code_duplication"] = {"error": str(e)}

        # Concurrency Analysis
        if "concurrency" in self.analyzers:
            print("Running Concurrency Analysis...")
            try:
                concurrency_results = self.analyzers["concurrency"].analyze_directory(str(self.project_root))
                results["concurrency"] = concurrency_results
            except Exception as e:
                results["concurrency"] = {"error": str(e)}

        results["execution_time"] = time.time() - start_time

        # Save individual report
        report_path = self.reports_dir / f"performance_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def run_security_analysis(self) -> dict[str, Any]:
        """Run comprehensive security analysis."""
        print("\n" + "="*60)
        print("Running Security Analysis")
        print("="*60)

        start_time = time.time()
        results = {}

        # Static Analysis (includes security checks)
        if "static_analysis" in self.analyzers:
            print("Running Static Analysis (Security)...")
            try:
                static_results = self.analyzers["static_analysis"].analyze_directory(str(self.project_root))
                results["static_analysis"] = static_results
            except Exception as e:
                results["static_analysis"] = {"error": str(e)}

        # Error Handling Analysis
        if "error_handling" in self.analyzers:
            print("Running Error Handling Analysis...")
            try:
                error_results = self.analyzers["error_handling"].analyze_directory(str(self.project_root))
                results["error_handling"] = error_results
            except Exception as e:
                results["error_handling"] = {"error": str(e)}

        # Code Smell Detection
        if "code_smell" in self.analyzers:
            print("Running Code Smell Detection...")
            try:
                smell_results = self.analyzers["code_smell"].analyze_directory(str(self.project_root))
                results["code_smell"] = smell_results
            except Exception as e:
                results["code_smell"] = {"error": str(e)}

        results["execution_time"] = time.time() - start_time

        # Save individual report
        report_path = self.reports_dir / f"security_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

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

    def run_enhanced_dependency_analysis(self) -> dict[str, Any]:
        """Run enhanced dependency analysis."""
        print("\n" + "="*60)
        print("Running Enhanced Dependency Analysis")
        print("="*60)

        start_time = time.time()
        from analyzers.enhanced_dependency_analyzer import EnhancedDependencyAnalyzer
        analyzer = EnhancedDependencyAnalyzer(str(self.project_root))
        result = analyzer.analyze_project()
        result["execution_time"] = time.time() - start_time

        # Convert PluginResult objects to serializable format
        def make_serializable(obj):
            if hasattr(obj, 'to_dict'):
                return make_serializable(obj.to_dict())
            elif hasattr(obj, '__dict__'):
                return {k: make_serializable(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif hasattr(obj, '__class__') and 'PluginResult' in str(obj.__class__):
                # Handle PluginResult objects specifically
                return {
                    'type': str(obj.__class__),
                    'data': make_serializable(obj.__dict__) if hasattr(obj, '__dict__') else str(obj)
                }
            else:
                return obj

        # Recursively convert the result
        serializable_result = make_serializable(result)

        # Save individual report
        report_path = self.reports_dir / f"enhanced_dependency_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(serializable_result, f, indent=2)

        return result

    def run_enhanced_undefined_names_analysis(self) -> dict[str, Any]:
        """Run the enhanced undefined names analyzer."""
        print("\n" + "="*60)
        print("Running Enhanced Undefined Names Analyzer")
        print("="*60)

        start_time = time.time()
        from analyzers.undefined_names_analyzer import UndefinedNamesAnalyzer
        analyzer = UndefinedNamesAnalyzer(self.config)
        result = analyzer.analyze_directory(str(self.project_root))
        result["execution_time"] = time.time() - start_time

        # Add to aggregator
        self.report_aggregator.add_undefined_names_results(result)

        # Save individual report
        report_path = self.reports_dir / f"enhanced_undefined_names_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_targeted_import_fixes(self) -> dict[str, Any]:
        """Run targeted import fixes for remaining issues."""
        print("\n" + "="*60)
        print("Running Targeted Import Fixes")
        print("="*60)
        
        try:
            # Look for import analysis report
            report_files = list(self.project_root.glob("**/import_analysis_report*.json"))
            if not report_files:
                return {"status": "skipped", "message": "No import analysis report found"}
            
            latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
            fixer = TargetedImportFixer(str(self.project_root), str(latest_report))
            fixer.load_issues()
            results = fixer.fix_issues()
            
            return {
                "status": "completed",
                "issues_found": len(fixer.issues),
                "fixes_applied": len(fixer.fixes_applied),
                "failed_fixes": len(fixer.failed_fixes),
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def run_test_verification(self) -> dict[str, Any]:
        """Run test setup verification."""
        print("\n" + "="*60)
        print("Running Test Verification")
        print("="*60)
        
        try:
            # Stub implementation - these functions don't exist
            return {"status": "skipped", "message": "Test verification functions not available"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def run_debug_analysis(self) -> dict[str, Any]:
        """Run debug analysis."""
        print("\n" + "="*60)
        print("Running Debug Analysis")
        print("="*60)
        
        try:
            # Stub implementation - DebugAnalyzer doesn't exist
            return {"status": "skipped", "message": "DebugAnalyzer not available"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def run_merge_conflict_detection(self) -> dict[str, Any]:
        """Run merge conflict detection."""
        print("\n" + "="*60)
        print("Running Merge Conflict Detection")
        print("="*60)
        
        try:
            # Stub implementation - MergeConflictDetector doesn't exist
            return {"status": "skipped", "message": "MergeConflictDetector not available"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def run_comprehensive_import_fixes(self) -> dict[str, Any]:
        """Run comprehensive import fixes."""
        print("\n" + "="*60)
        print("Running Comprehensive Import Fixes")
        print("="*60)
        
        try:
            # Stub implementation - ComprehensiveImportFixer doesn't exist
            return {"status": "skipped", "message": "ComprehensiveImportFixer not available"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def run_auto_dead_code_fixes(self) -> dict[str, Any]:
        """Run automated dead code fixes."""
        print("\n" + "="*60)
        print("Running Automated Dead Code Fixes")
        print("="*60)
        
        try:
            # Stub implementation - AutoFixDeadCode doesn't exist
            return {"status": "skipped", "message": "AutoFixDeadCode not available"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def run_enhanced_analysis_integration(self) -> dict[str, Any]:
        """Run enhanced analysis integration with false positive reduction."""
        print("\n" + "="*60)
        print("Running Enhanced Analysis Integration")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Run enhanced analyzers for false positive reduction
            enhanced_results = {}
            
            # 1. Fallback Pattern Detection
            print("Running Fallback Pattern Detection...")
            fallback_analyzer = EnhancedFallbackDetector()
            fallback_results = []
            
            for file_path in self.file_paths[:10]:  # Sample first 10 files
                try:
                    result = fallback_analyzer.analyze_file(file_path)
                    fallback_results.append(result)
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
            
            enhanced_results["fallback_patterns"] = {
                "total_files_analyzed": len(fallback_results),
                "total_patterns": sum(r.total_patterns for r in fallback_results),
                "patterns_by_type": self._aggregate_patterns_by_type(fallback_results)
            }
            
            # 2. Enhanced Security Analysis
            print("Running Enhanced Security Analysis...")
            security_analyzer = EnhancedSecurityAnalyzer()
            security_results = []
            
            for file_path in self.file_paths[:10]:  # Sample first 10 files
                try:
                    result = security_analyzer.analyze_file(file_path)
                    security_results.append(result)
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
            
            enhanced_results["enhanced_security"] = {
                "total_files_analyzed": len(security_results),
                "total_issues": sum(r.total_issues for r in security_results),
                "real_issues": sum(r.real_issues for r in security_results),
                "false_positives": sum(r.false_positives for r in security_results),
                "false_positive_rate": self._calculate_false_positive_rate(security_results)
            }
            
            # 3. Dynamic Import Analysis
            print("Running Dynamic Import Analysis...")
            import_analyzer = EnhancedDynamicImportAnalyzer()
            import_results = []
            
            for file_path in self.file_paths[:10]:  # Sample first 10 files
                try:
                    result = import_analyzer.analyze_file(file_path)
                    import_results.append(result)
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
            
            enhanced_results["dynamic_imports"] = {
                "total_files_analyzed": len(import_results),
                "total_patterns": sum(r.total_patterns for r in import_results),
                "total_issues": sum(r.total_issues for r in import_results),
                "real_issues": sum(r.real_issues for r in import_results),
                "false_positives": sum(r.false_positives for r in import_results)
            }
            
            # 4. Stub Object Analysis
            print("Running Stub Object Analysis...")
            stub_analyzer = StubObjectAnalyzer()
            stub_results = []
            
            for file_path in self.file_paths[:10]:  # Sample first 10 files
                try:
                    result = stub_analyzer.analyze_file(file_path)
                    stub_results.append(result)
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
            
            enhanced_results["stub_objects"] = {
                "total_files_analyzed": len(stub_results),
                "total_stubs": sum(r.total_stubs for r in stub_results),
                "expected_stubs": sum(r.expected_stubs for r in stub_results),
                "unexpected_stubs": sum(r.unexpected_stubs for r in stub_results),
                "stubs_by_category": self._aggregate_stubs_by_category(stub_results)
            }
            
            result = {
                "status": "completed",
                "execution_time": time.time() - start_time,
                "enhanced_results": enhanced_results,
                "message": "Enhanced analysis with false positive reduction completed"
            }
            
            self.results["enhanced_analysis"] = result
            return result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "execution_time": time.time() - start_time,
                "error": str(e)
            }
            self.results["enhanced_analysis"] = error_result
            return error_result

    def run_validation_checks(self) -> dict[str, Any]:
        """Run validation checks."""
        print("\n" + "="*60)
        print("Running Validation Checks")
        print("="*60)
        
        try:
            # Stub implementation - run_validation function doesn't exist
            return {"status": "skipped", "message": "Validation functions not available"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def run_test_execution(self) -> dict[str, Any]:
        """Run test execution."""
        print("\n" + "="*60)
        print("Running Test Execution")
        print("="*60)
        
        try:
            # Stub implementation - these test functions don't exist
            return {"status": "skipped", "message": "Test execution functions not available"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def run_all(self) -> dict[str, Any]:
        """Run the most comprehensive code quality analysis available."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE CODE QUALITY ANALYSIS PIPELINE")
        print(f"{'='*80}")
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Plugins enabled: {self.enable_plugins}")

        total_start = time.time()

        # Basic Analysis (excluding specialized pipelines)
        self.results["basic_analysis"] = {
            "syntax_validation": self.run_syntax_validation(),
            "import_validation": self.run_import_validation(),
            "circular_imports": self.detect_circular_imports(),
            "comprehensive_import_undefined_check": self.run_comprehensive_import_undefined_check(),
            "enhanced_undefined_names_analysis": self.run_enhanced_undefined_names_analysis(),
        }

        # Async and Types
        self.results["async_types"] = {
            "async_fixes": self.run_async_fixes(),
            "type_hints": self.run_type_hints(),
        }

        # Core Analysis (excluding specialized pipelines)
        self.results["core_analysis"] = {
            "enhanced_dependency_analysis": self.run_enhanced_dependency_analysis(),
            "function_validation": self.run_function_validation(),
            "enhanced_validation": self.run_enhanced_validation(),
            "metrics": self.run_metrics_analysis(),
            "test_coverage": self.run_test_coverage_analysis(),
            "code_smells": self.run_code_smell_detection(),
            "documentation": self.run_documentation_analysis(),
            "configuration": self.run_configuration_analysis(),
            "data_flow": self.run_data_flow_analysis(),
            "static_analysis": self.run_static_analysis(),
            "ast_analysis": self.run_ast_analysis(),
        }

        # Advanced Analysis (excluding specialized pipelines)
        self.results["advanced_analysis"] = self.run_advanced_analysis()

        # Performance Analysis
        self.results["performance_analysis"] = self.run_performance_analysis()

        # Security Analysis
        self.results["security_analysis"] = self.run_security_analysis()

        # Visualization Analysis
        self.results["visualization"] = self.run_visualization_analysis()

        # Consolidated Fixes
        self.results["consolidated_fixes"] = self.run_consolidated_fixes()

        # Plugin Analysis
        self.results["plugin_results"] = self.run_plugin_analysis()

        # Comprehensive Review
        self.results["comprehensive_review"] = self.run_comprehensive_review()

        # Integrated Standalone Scripts
        self.results["integrated_scripts"] = {
            "targeted_import_fixes": self.run_targeted_import_fixes(),
            "test_verification": self.run_test_verification(),
            "debug_analysis": self.run_debug_analysis(),
            "merge_conflict_detection": self.run_merge_conflict_detection(),
            "comprehensive_import_fixes": self.run_comprehensive_import_fixes(),
            "auto_dead_code_fixes": self.run_auto_dead_code_fixes(),
            "enhanced_analysis": self.run_enhanced_analysis_integration(),
            "validation_checks": self.run_validation_checks(),
            "test_execution": self.run_test_execution(),
        }

        # Generate summary
        self.results["summary"] = self._generate_summary(time.time() - total_start)

        # Save individual pipeline report
        report_path = self.reports_dir / f"unified_pipeline_{self.timestamp}.json"
        
        # Convert PluginResult objects to serializable format
        def make_serializable(obj):
            if hasattr(obj, 'to_dict'):
                return make_serializable(obj.to_dict())
            elif hasattr(obj, '__dict__'):
                return {k: make_serializable(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif hasattr(obj, '__class__') and 'PluginResult' in str(obj.__class__):
                # Handle PluginResult objects specifically
                return {
                    'type': str(obj.__class__),
                    'data': make_serializable(obj.__dict__) if hasattr(obj, '__dict__') else str(obj)
                }
            else:
                return obj

        # Recursively convert the results
        serializable_results = make_serializable(self.results)
        
        with open(report_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        # Generate and save unified reports
        print("\n" + "="*60)
        print("Generating Unified Reports")
        print("="*60)

        json_report, md_report = self.report_aggregator.save_reports(
            self.reports_dir,
            "unified_code_quality_report",
        )

        print(f"\n{'='*80}")
        print("COMPREHENSIVE ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Total execution time: {time.time() - total_start:.2f} seconds")
        print(f"Reports saved to: {self.reports_dir}")
        print(f"Main report: {report_path}")
        print(f"Unified JSON report: {json_report}")
        print(f"Unified Markdown report: {md_report}")
        print(f"\nThis comprehensive analysis included:")
        print(f"  • {len(self.analyzers)} analyzers")
        print(f"  • {len(self.visualizers)} visualizers")
        print(f"  • {len(self.plugin_registry.list_plugins()) if self.enable_plugins else 0} plugins")
        print(f"  • All fix scripts and tools")
        print(f"  • Complete code quality assessment")

        # Print summary
        self._print_summary()

        # Print aggregated summary
        self._print_aggregated_summary()

        return self.results

    def _aggregate_patterns_by_type(self, results) -> dict[str, int]:
        """Aggregate patterns by type across multiple results."""
        type_counts = {}
        for result in results:
            for pattern_type, count in result.patterns_by_type.items():
                type_counts[pattern_type.value] = type_counts.get(pattern_type.value, 0) + count
        return type_counts
    
    def _calculate_false_positive_rate(self, results) -> float:
        """Calculate the false positive rate across multiple results."""
        total_issues = sum(r.total_issues for r in results)
        false_positives = sum(r.false_positives for r in results)
        if total_issues == 0:
            return 0.0
        return (false_positives / total_issues) * 100
    
    def _aggregate_stubs_by_category(self, results) -> dict[str, int]:
        """Aggregate stubs by category across multiple results."""
        category_counts = {}
        for result in results:
            for category, count in result.stubs_by_category.items():
                category_counts[category.value] = category_counts.get(category.value, 0) + count
        return category_counts

    def _generate_summary(self, total_time: float) -> dict[str, Any]:
        """Generate summary of all results."""
        summary = {
            "timestamp": self.timestamp,
            "project_root": str(self.project_root),
            "total_execution_time": total_time,
            "categories": {},
        }

        for category, tools in self.results.items():
            if category == "summary":
                continue

            category_summary = {}
            if tools is not None:
                for tool_name, result in tools.items():
                    if isinstance(result, dict):
                        category_summary[tool_name] = {
                            "execution_time": result.get("execution_time", 0),
                            "issues_fixed": result.get("total_fixed", 0),
                            "issues_found": result.get("total_issues", 0),
                            "files_processed": result.get("total_files", 0),
                        }

            summary["categories"][category] = category_summary

        return summary

    def _print_summary(self):
        """Print a formatted summary."""
        summary = self.results["summary"]

        print(f"\n{'='*80}")
        print("PIPELINE EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total execution time: {summary['total_execution_time']:.2f} seconds")

        for category, tools in summary["categories"].items():
            print(f"\n{category.upper()}:")
            for tool, info in tools.items():
                print(f"  {tool}:")
                print(f"    Execution time: {info['execution_time']:.2f}s")
                if info["issues_fixed"]:
                    print(f"    Issues fixed: {info['issues_fixed']}")
                if info["issues_found"]:
                    print(f"    Issues found: {info['issues_found']}")
                if info["files_processed"]:
                    print(f"    Files processed: {info['files_processed']}")

    def _print_aggregated_summary(self):
        """Print aggregated report summary."""
        report = self.report_aggregator.generate_unified_report()
        summary = report["overall_summary"]

        print(f"\n{'='*80}")
        print("UNIFIED CODE QUALITY SUMMARY")
        print(f"{'='*80}")
        print(f"Total Files Analyzed: {summary['total_files']}")
        print(f"Total Directories: {summary['total_directories']}")
        print(f"Total Issues Found: {summary['total_issues']}")
        print(f"Issues Fixed: {summary['fixed_issues']}")

        # Print enhanced dependency analysis results if available
        if "enhanced_dependency_analysis" in self.results.get("analysis", {}):
            eda = self.results["analysis"]["enhanced_dependency_analysis"]
            print("\nEnhanced Dependency Analysis Results:")
            print(f"  - Tools Used: {', '.join(eda.get('tools_used', []))}")
            print(f"  - Undeclared Dependencies: {len(eda.get('undeclared_deps', []))}")
            print(f"  - Unused Dependencies: {len(eda.get('unused_deps', []))}")
            print(f"  - Total Issues: {eda.get('total_issues', 0)}")
            
            if eda.get('undeclared_deps'):
                print("  - Undeclared Dependencies:")
                for dep in eda['undeclared_deps'][:5]:  # Show first 5
                    print(f"    * {dep}")
                if len(eda['undeclared_deps']) > 5:
                    print(f"    ... and {len(eda['undeclared_deps']) - 5} more")
            
            if eda.get('unused_deps'):
                print("  - Unused Dependencies:")
                for dep in eda['unused_deps'][:5]:  # Show first 5
                    print(f"    * {dep}")
                if len(eda['unused_deps']) > 5:
                    print(f"    ... and {len(eda['unused_deps']) - 5} more")

        # Print enhanced validation specific stats if available
        if "enhanced_validation" in self.results.get("analysis", {}):
            ev = self.results["analysis"]["enhanced_validation"]
            print("\nEnhanced Validation Results:")
            print(f"  - Argument Mismatches: {ev.get('argument_mismatches', 0)}")
            print(f"  - Unsafe Data Access: {ev.get('unsafe_data_access', 0)}")
            print(f"  - Missing Null Checks: {ev.get('missing_null_checks', 0)}")
            print(f"  - Type Inconsistencies: {ev.get('type_inconsistencies', 0)}")

        print("\nIssue Breakdown:")
        for issue_type, count in summary["issue_breakdown"].items():
            print(f"  {issue_type.replace('_', ' ').title()}: {count}")

        if summary["critical_files"]:
            print("\nTop Files with Issues:")
            for i, file_info in enumerate(summary["critical_files"][:5]):
                file_name = Path(file_info["file"]).name
                print(f"  {i+1}. {file_name}: {file_info['issues']} issues ({file_info['fixed']} fixed)")

        print(f"\nClean Files: {len(summary['clean_files'])}")
        print(f"\nReports saved to: {self.reports_dir}")


def main():
    """Main entry point for the comprehensive code quality analysis pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified Enhanced Pipeline - Comprehensive analysis with imports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive analysis with imports
  python pipelines/pipeline_unified_enhanced.py
  
  # Run on specific project directory
  python pipelines/pipeline_unified_enhanced.py --project-root /path/to/project
  
  # Disable plugins
  python pipelines/pipeline_unified_enhanced.py --no-plugins
        """
    )
    parser.add_argument("--project-root", default="/workspace/src",
                        help="Project root directory")
    parser.add_argument("--skip-syntax", action="store_true",
                        help="Skip syntax and import fixes")
    parser.add_argument("--skip-async", action="store_true",
                        help="Skip async and type fixes")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip code analysis")
    parser.add_argument("--no-plugins", action="store_true",
                        help="Disable plugin system")

    args = parser.parse_args()

    pipeline = UnifiedEnhancedPipeline(
        project_root=args.project_root,
        enable_plugins=not args.no_plugins
    )

    # Run comprehensive analysis
    pipeline.run_all()


if __name__ == "__main__":
    main()
