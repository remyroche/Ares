#!/usr/bin/env python3
"""
Unified Code Quality Pipeline - Enhanced Version with Dependency Management

This version provides comprehensive unified reporting with per-file and
per-directory information using the ReportAggregator, with improved
dependency handling and reduced redundancy.
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

# Import base pipeline and dependency manager
from pipelines.base_pipeline import BasePipeline
from utils.dependency_manager import dependency_manager, safe_import

# Safe imports with fallbacks
try:
    from analyzers.code_smell_detector import CodeSmellDetector
except ImportError:
    CodeSmellDetector = None

try:
    from analyzers.configuration_analyzer import ConfigurationAnalyzer
except ImportError:
    ConfigurationAnalyzer = None

try:
    from analyzers.data_flow_analyzer import DataFlowAnalyzer
except ImportError:
    DataFlowAnalyzer = None

try:
    from analyzers.documentation_analyzer import DocumentationAnalyzer
except ImportError:
    DocumentationAnalyzer = None

try:
    from analyzers.metrics_analyzer import MetricsAnalyzer
except ImportError:
    MetricsAnalyzer = None

try:
    from analyzers.performance_analyzer import PerformanceAnalyzer
except ImportError:
    PerformanceAnalyzer = None

try:
    from analyzers.test_coverage_analyzer import TestCoverageAnalyzer
except ImportError:
    TestCoverageAnalyzer = None

try:
    from comprehensive_code_review import CodeQualityReviewer
except ImportError:
    CodeQualityReviewer = None

try:
    from enhanced_validator import EnhancedValidator
except ImportError:
    EnhancedValidator = None

try:
    from function_validator import FunctionValidator
except ImportError:
    FunctionValidator = None

# Script imports with fallbacks
try:
    from scripts.advanced_syntax_fixer import AdvancedSyntaxFixer
except ImportError:
    AdvancedSyntaxFixer = None

try:
    from scripts.detect_circular_imports import ImportAnalyzer as CircularImportDetector
except ImportError:
    CircularImportDetector = None

try:
    from scripts.enhanced_type_hints import TypeHintEnhancer
except ImportError:
    TypeHintEnhancer = None

try:
    from scripts.robust_async_fixer import RobustAsyncFixer
except ImportError:
    RobustAsyncFixer = None

try:
    from scripts.safe_import_fixer import SafeImportFixer
except ImportError:
    SafeImportFixer = None

try:
    from scripts.simple_interaction_mapper import extract_interactions
except ImportError:
    extract_interactions = None

try:
    from utils.report_aggregator import ReportAggregator
except ImportError:
    ReportAggregator = None


class UnifiedEnhancedPipeline(BasePipeline):
    """Enhanced unified pipeline with comprehensive reporting and dependency management."""

    def __init__(self, project_root: str = "/workspace/src"):
        super().__init__(project_root)
        self.report_aggregator = ReportAggregator(project_root) if ReportAggregator else None
        
        # Print dependency status
        dependency_manager.print_dependency_status()

    def run_syntax_fixes(self) -> dict[str, Any]:
        """Run advanced syntax fixes."""
        self._print_section_header("Running Advanced Syntax Fixes")

        if not AdvancedSyntaxFixer:
            return self._handle_error(
                Exception("AdvancedSyntaxFixer not available - missing dependencies"),
                "syntax_fixes"
            )

        start_time = time.time()
        try:
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

            # Add to aggregator if available
            if self.report_aggregator:
                self.report_aggregator.add_syntax_results(result)

            # Save individual report
            self._save_report(result, "syntax_fixes")
            return result

        except Exception as e:
            return self._handle_error(e, "syntax_fixes")

    def run_import_fixes(self) -> dict[str, Any]:
        """Run import fixes."""
        self._print_section_header("Running Import Fixes")

        if not SafeImportFixer:
            return self._handle_error(
                Exception("SafeImportFixer not available - missing dependencies"),
                "import_fixes"
            )

        start_time = time.time()
        try:
            fixer = SafeImportFixer(str(self.project_root))
            fixer.fix_project(dry_run=False)

            # Convert fixed_files to proper format for report aggregator
            fixed_files_formatted = []
            for file_path in fixer.fixed_files:
                fixed_files_formatted.append({
                    "file": file_path,
                    "imports_added": []  # SafeImportFixer doesn't track specific imports added
                })
            
            result = {
                "fixed_files": fixed_files_formatted,
                "failed_files": fixer.failed_files,
                "import_errors": {},  # SafeImportFixer doesn't have import_errors attribute
                "total_fixed": len(fixer.fixed_files),
                "total_failed": len(fixer.failed_files),
                "execution_time": time.time() - start_time,
            }

            # Add to aggregator if available
            if self.report_aggregator:
                self.report_aggregator.add_import_results(result)

            # Save individual report
            self._save_report(result, "import_fixes")
            return result

        except Exception as e:
            return self._handle_error(e, "import_fixes")

    def detect_circular_imports(self) -> dict[str, Any]:
        """Detect circular imports."""
        self._print_section_header("Detecting Circular Imports")

        if not CircularImportDetector:
            return self._handle_error(
                Exception("CircularImportDetector not available - missing dependencies"),
                "circular_imports"
            )

        start_time = time.time()
        try:
            detector = CircularImportDetector(str(self.project_root))
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

            # Add to aggregator if available
            if self.report_aggregator:
                self.report_aggregator.add_circular_import_results(result)

            # Save individual report
            self._save_report(result, "circular_imports")
            return result

        except Exception as e:
            return self._handle_error(e, "circular_imports")

    def run_async_fixes(self) -> dict[str, Any]:
        """Run robust async/await fixes."""
        self._print_section_header("Running Async/Await Fixes")

        if not RobustAsyncFixer:
            return self._handle_error(
                Exception("RobustAsyncFixer not available - missing dependencies"),
                "async_fixes"
            )

        start_time = time.time()
        try:
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

            # Add to aggregator if available
            if self.report_aggregator:
                self.report_aggregator.add_async_results(result)

            # Save individual report
            self._save_report(result, "async_fixes")
            return result

        except Exception as e:
            return self._handle_error(e, "async_fixes")

    def run_type_hints(self) -> dict[str, Any]:
        """Run type hint enhancements."""
        self._print_section_header("Running Type Hint Enhancements")

        if not TypeHintEnhancer:
            return self._handle_error(
                Exception("TypeHintEnhancer not available - missing dependencies"),
                "type_hints"
            )

        start_time = time.time()
        try:
            # Get all Python files
            python_files = self._find_python_files()

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

            # Add to aggregator if available
            if self.report_aggregator:
                self.report_aggregator.add_type_results(result)

            # Save individual report
            self._save_report(result, "type_hints")
            return result

        except Exception as e:
            return self._handle_error(e, "type_hints")

    def run_function_validation(self) -> dict[str, Any]:
        """Run function validation checks."""
        self._print_section_header("Running Function Validation")

        if not FunctionValidator:
            return self._handle_error(
                Exception("FunctionValidator not available - missing dependencies"),
                "function_validation"
            )

        start_time = time.time()
        try:
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

            # Add to aggregator if available
            if self.report_aggregator:
                self.report_aggregator.add_function_validation_results(result)

            # Save individual report
            self._save_report(result, "function_validation")
            return result

        except Exception as e:
            return self._handle_error(e, "function_validation")

    def run_enhanced_validation(self) -> dict[str, Any]:
        """Run enhanced validation for function arguments and data access."""
        self._print_section_header("Running Enhanced Validation (Arguments & Data Access)")

        if not EnhancedValidator:
            return self._handle_error(
                Exception("EnhancedValidator not available - missing dependencies"),
                "enhanced_validation"
            )

        start_time = time.time()
        try:
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

            # Add to aggregator if available
            if self.report_aggregator:
                self.report_aggregator.add_enhanced_validation_results(report)

            # Save individual report
            self._save_report(result, "enhanced_validation")
            return result

        except Exception as e:
            return self._handle_error(e, "enhanced_validation")

    def run_comprehensive_review(self) -> dict[str, Any]:
        """Run comprehensive code quality review."""
        self._print_section_header("Running Comprehensive Code Review")

        if not CodeQualityReviewer:
            return self._handle_error(
                Exception("CodeQualityReviewer not available - missing dependencies"),
                "comprehensive_review"
            )

        start_time = time.time()
        try:
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

            # Add to aggregator if available
            if self.report_aggregator:
                self.report_aggregator.add_comprehensive_review_results(result)

            # Save individual report
            self._save_report(result, "comprehensive_review")
            return result

        except Exception as e:
            return self._handle_error(e, "comprehensive_review")

    def run_interaction_mapping(self) -> dict[str, Any]:
        """Run code interaction mapping."""
        self._print_section_header("Running Code Interaction Mapping")

        if not extract_interactions:
            return self._handle_error(
                Exception("extract_interactions not available - missing dependencies"),
                "interaction_mapping"
            )

        start_time = time.time()
        try:
            # Use the comprehensive review data
            if CodeQualityReviewer:
                reviewer = CodeQualityReviewer(str(self.project_root))
                reviewer.review_directory(str(self.project_root))
                report_data = reviewer.generate_report()

                # Extract interactions
                interactions = extract_interactions(report_data)

                result = {
                    "interactions": interactions,
                    "module_count": len(interactions["import_graph"]),
                    "function_count": len(interactions["function_definitions"]),
                    "undefined_functions": len(interactions["undefined_functions"]),
                    "async_issues": len(interactions["async_patterns"]),
                    "execution_time": time.time() - start_time,
                }

                # Save reports
                self._save_report(result, "code_interactions")
                return result
            else:
                return self._handle_error(
                    Exception("CodeQualityReviewer not available for interaction mapping"),
                    "interaction_mapping"
                )

        except Exception as e:
            return self._handle_error(e, "interaction_mapping")

    def run_metrics_analysis(self) -> dict[str, Any]:
        """Run code metrics analysis."""
        self._print_section_header("Running Code Metrics Analysis")

        if not MetricsAnalyzer:
            return self._handle_error(
                Exception("MetricsAnalyzer not available - missing dependencies"),
                "metrics_analysis"
            )

        start_time = time.time()
        try:
            analyzer = MetricsAnalyzer(str(self.project_root))

            # Analyze all Python files
            python_files = self._find_python_files()
            for file_path in python_files:
                analyzer.analyze_file(file_path)

            result = analyzer.generate_report()
            result["execution_time"] = time.time() - start_time

            # Add to aggregator if available
            if self.report_aggregator:
                self.report_aggregator.file_metrics.update(analyzer.file_metrics)

            # Save report
            self._save_report(result, "metrics_analysis")
            return result

        except Exception as e:
            return self._handle_error(e, "metrics_analysis")

    def run_test_coverage_analysis(self) -> dict[str, Any]:
        """Run test coverage analysis."""
        self._print_section_header("Running Test Coverage Analysis")

        if not TestCoverageAnalyzer:
            return self._handle_error(
                Exception("TestCoverageAnalyzer not available - missing dependencies"),
                "test_coverage"
            )

        start_time = time.time()
        try:
            analyzer = TestCoverageAnalyzer(str(self.project_root))
            result = analyzer.analyze_project()
            result["execution_time"] = time.time() - start_time

            # Save report
            self._save_report(result, "test_coverage")
            return result

        except Exception as e:
            return self._handle_error(e, "test_coverage")

    def run_code_smell_detection(self) -> dict[str, Any]:
        """Run code smell detection."""
        self._print_section_header("Running Code Smell Detection")

        if not CodeSmellDetector:
            return self._handle_error(
                Exception("CodeSmellDetector not available - missing dependencies"),
                "code_smells"
            )

        start_time = time.time()
        try:
            detector = CodeSmellDetector(str(self.project_root))

            # Analyze all Python files
            python_files = self._find_python_files()
            for file_path in python_files:
                detector.analyze_file(file_path)

            result = detector.generate_report()
            result["execution_time"] = time.time() - start_time

            # Save report
            self._save_report(result, "code_smells")
            return result

        except Exception as e:
            return self._handle_error(e, "code_smells")

    def run_documentation_analysis(self) -> dict[str, Any]:
        """Run documentation quality analysis."""
        self._print_section_header("Running Documentation Analysis")

        if not DocumentationAnalyzer:
            return self._handle_error(
                Exception("DocumentationAnalyzer not available - missing dependencies"),
                "documentation"
            )

        start_time = time.time()
        try:
            analyzer = DocumentationAnalyzer(str(self.project_root))

            # Analyze all Python files
            python_files = self._find_python_files()
            for file_path in python_files:
                analyzer.analyze_file(file_path)

            # Analyze README
            analyzer.analyze_readme()

            result = analyzer.generate_report()
            result["execution_time"] = time.time() - start_time

            # Save report
            self._save_report(result, "documentation_analysis")
            return result

        except Exception as e:
            return self._handle_error(e, "documentation")

    def run_performance_analysis(self) -> dict[str, Any]:
        """Run performance analysis."""
        self._print_section_header("Running Performance Analysis")

        if not PerformanceAnalyzer:
            return self._handle_error(
                Exception("PerformanceAnalyzer not available - missing dependencies"),
                "performance"
            )

        start_time = time.time()
        try:
            analyzer = PerformanceAnalyzer(str(self.project_root))

            # Analyze all Python files
            python_files = self._find_python_files()
            for file_path in python_files:
                analyzer.analyze_file(file_path)

            result = analyzer.generate_report()
            result["execution_time"] = time.time() - start_time

            # Save report
            self._save_report(result, "performance_analysis")
            return result

        except Exception as e:
            return self._handle_error(e, "performance")

    def run_configuration_analysis(self) -> dict[str, Any]:
        """Run configuration analysis."""
        self._print_section_header("Running Configuration Analysis")

        if not ConfigurationAnalyzer:
            return self._handle_error(
                Exception("ConfigurationAnalyzer not available - missing dependencies"),
                "configuration"
            )

        start_time = time.time()
        try:
            analyzer = ConfigurationAnalyzer(str(self.project_root))
            result = analyzer.analyze_project()
            result["execution_time"] = time.time() - start_time

            # Save report
            self._save_report(result, "configuration_analysis")
            return result

        except Exception as e:
            return self._handle_error(e, "configuration")

    def run_data_flow_analysis(self) -> dict[str, Any]:
        """Run data flow analysis."""
        self._print_section_header("Running Data Flow Analysis")

        if not DataFlowAnalyzer:
            return self._handle_error(
                Exception("DataFlowAnalyzer not available - missing dependencies"),
                "data_flow"
            )

        start_time = time.time()
        try:
            analyzer = DataFlowAnalyzer(str(self.project_root))

            # Analyze all Python files
            python_files = self._find_python_files()
            for file_path in python_files:
                analyzer.analyze_file(file_path)

            result = analyzer.generate_report()
            result["execution_time"] = time.time() - start_time

            # Save report
            self._save_report(result, "data_flow_analysis")
            return result

        except Exception as e:
            return self._handle_error(e, "data_flow")

    def run_all(self) -> dict[str, Any]:
        """Run all code quality tools with unified reporting."""
        self._print_pipeline_header("UNIFIED CODE QUALITY PIPELINE - ENHANCED (FIXED)")

        # Validate project root
        if not self._validate_project_root():
            print("Warning: Project root validation failed, but continuing...")

        # Syntax and Imports
        self.results["syntax_imports"] = {
            "syntax_fixes": self.run_syntax_fixes(),
            "import_fixes": self.run_import_fixes(),
            "circular_imports": self.detect_circular_imports(),
        }

        # Async and Types
        self.results["async_types"] = {
            "async_fixes": self.run_async_fixes(),
            "type_hints": self.run_type_hints(),
        }

        # Analysis
        self.results["analysis"] = {
            "function_validation": self.run_function_validation(),
            "enhanced_validation": self.run_enhanced_validation(),
            "comprehensive_review": self.run_comprehensive_review(),
            "interaction_mapping": self.run_interaction_mapping(),
            "metrics": self.run_metrics_analysis(),
            "test_coverage": self.run_test_coverage_analysis(),
            "code_smells": self.run_code_smell_detection(),
            "documentation": self.run_documentation_analysis(),
            "performance": self.run_performance_analysis(),
            "configuration": self.run_configuration_analysis(),
            "data_flow": self.run_data_flow_analysis(),
        }

        # Generate summary
        total_time = self._finalize_execution_tracking()
        self.results["summary"] = self._generate_summary(total_time)

        # Save individual pipeline report
        self._save_report(self.results, "unified_pipeline")

        # Generate and save unified reports if aggregator is available
        if self.report_aggregator:
            self._print_section_header("Generating Unified Reports")
            try:
                json_report, md_report = self.report_aggregator.save_reports(
                    self.reports_dir,
                    "unified_code_quality_report",
                )
                print("\nUnified reports saved:")
                print(f"  JSON: {json_report}")
                print(f"  Markdown: {md_report}")
            except Exception as e:
                print(f"Warning: Failed to generate unified reports: {e}")

        # Print summary
        self._print_summary(self.results["summary"])

        return self.results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run unified code quality pipeline with enhanced reporting and dependency management",
    )
    parser.add_argument("--project-root", default="/workspace/src",
                        help="Project root directory")
    parser.add_argument("--skip-syntax", action="store_true",
                        help="Skip syntax and import fixes")
    parser.add_argument("--skip-async", action="store_true",
                        help="Skip async and type fixes")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip code analysis")

    args = parser.parse_args()

    with UnifiedEnhancedPipeline(args.project_root) as pipeline:
        # You could implement selective running based on args
        # For now, just run all
        pipeline.run_all()


if __name__ == "__main__":
    main()