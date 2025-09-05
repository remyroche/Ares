#!/usr/bin/env python3
"""
Simplified Code Quality Pipeline - Working Version

This is a simplified version of the code quality analysis pipeline that only uses
available modules and focuses on core functionality.
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

# Import available analyzers
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
from analyzers.code_smell_detector import CodeSmellDetector
from analyzers.configuration_analyzer import ConfigurationAnalyzer
from analyzers.documentation_analyzer import DocumentationAnalyzer
from analyzers.performance_analyzer import PerformanceAnalyzer
from analyzers.test_coverage_analyzer import TestCoverageAnalyzer
from analyzers.concurrency_analyzer import ConcurrencyAnalyzer
from analyzers.error_handling_analyzer import ErrorHandlingAnalyzer
from analyzers.code_duplication_analyzer import CodeDuplicationAnalyzer
from analyzers.improved_signature_analyzer import ImprovedSignatureAnalyzer

# Import available validators
from validators.enhanced_validator import EnhancedValidator
from validators.function_validator import FunctionValidator

# Import available scripts
from scripts.robust_async_fixer import RobustAsyncFixer
from scripts.advanced_syntax_fixer import AdvancedSyntaxFixer
from scripts.fix_missing_imports import ImportFixer
from scripts.enhanced_type_hints import TypeHintEnhancer

# Import core components
from core.config import get_default_config

# Import utilities
from utils.file_utils import find_python_files, get_directory_stats
from utils.gitignore_parser import filter_ignored_files


class SimplifiedPipeline:
    """
    Simplified Code Quality Analysis Pipeline
    
    This pipeline focuses on core functionality using only available modules.
    """

    def __init__(self, project_root: str = "/workspace/src"):
        self.project_root = Path(project_root)
        self.reports_dir = Path("/workspace/code_quality/reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config = get_default_config()
        self.results = {}
        
        # Get all Python files
        self.file_paths = find_python_files(str(self.project_root))
        
        # Initialize analyzers
        self._initialize_analyzers()

    def _initialize_analyzers(self):
        """Initialize available analyzers."""
        try:
            self.analyzers = {
                "syntax_validator": SyntaxValidator(self.config),
                "linter": LinterAnalyzer(self.config),
                "undefined_names": UndefinedNamesAnalyzer(self.config),
                "type_checker": TypeChecker(self.config),
                "static_analysis": StaticAnalysisAnalyzer(self.config),
                "ast_analysis": ASTAnalysisAnalyzer(self.config),
                "architecture": ArchitectureAnalyzer(self.config),
                "call_graph": CallGraphAnalyzer(self.config),
                "complexity": ComplexityAnalyzer(self.config),
                "dependency": DependencyAnalyzer(self.config),
                "metrics": MetricsAnalyzer(self.config),
                "data_flow": DataFlowAnalyzer(self.config),
                "enhanced_dead_code": EnhancedDeadCodeAnalyzer(self.config),
                "code_smell": CodeSmellDetector(self.config),
                "configuration": ConfigurationAnalyzer(self.config),
                "documentation": DocumentationAnalyzer(self.config),
                "performance": PerformanceAnalyzer(self.config),
                "test_coverage": TestCoverageAnalyzer(self.config),
                "concurrency": ConcurrencyAnalyzer(self.config),
                "error_handling": ErrorHandlingAnalyzer(self.config),
                "code_duplication": CodeDuplicationAnalyzer(self.config),
                "improved_signature": ImprovedSignatureAnalyzer(self.config),
            }
            print(f"✓ {len(self.analyzers)} analyzers initialized successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize some analyzers: {e}")
            self.analyzers = {}

    def run_syntax_validation(self) -> dict[str, Any]:
        """Run syntax validation."""
        print("\n" + "="*60)
        print("Running Syntax Validation")
        print("="*60)
        
        try:
            results = self.analyzers["syntax_validator"].validate_syntax(str(self.project_root))
            return {"status": "completed", "results": results}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def run_import_validation(self) -> dict[str, Any]:
        """Run import validation."""
        print("\n" + "="*60)
        print("Running Import Validation")
        print("="*60)
        
        try:
            results = self.analyzers["undefined_names"].analyze_directory(str(self.project_root))
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

        # Save individual report
        report_path = self.reports_dir / f"syntax_fixes_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_import_fixes(self) -> dict[str, Any]:
        """Run import fixes."""
        print("\n" + "="*60)
        print("Running Import Fixes")
        print("="*60)

        start_time = time.time()
        
        try:
            fixer = ImportFixer(str(self.project_root))
            fixer.fix_project(dry_run=False)
            
            result = {
                "fixed_files": fixer.fixed_files,
                "failed_files": fixer.failed_files,
                "total_fixed": len(fixer.fixed_files),
                "total_failed": len(fixer.failed_files),
                "execution_time": time.time() - start_time,
            }
        except Exception as e:
            result = {
                "fixed_files": [],
                "failed_files": [],
                "total_fixed": 0,
                "total_failed": 0,
                "execution_time": time.time() - start_time,
                "error": str(e)
            }

        # Save individual report
        report_path = self.reports_dir / f"import_fixes_{self.timestamp}.json"
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
        
        result = {
            "fixed_files": fixer.fixed_files,
            "failed_files": fixer.failed_files,
            "total_fixed": len(fixer.fixed_files),
            "total_failed": len(fixer.failed_files),
            "execution_time": time.time() - start_time,
        }

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
        fixed_files = []
        failed_files = []

        for file_path in self.file_paths[:10]:  # Limit to first 10 files for testing
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

        # Save individual report
        report_path = self.reports_dir / f"type_hints_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_analysis(self, analyzer_name: str) -> dict[str, Any]:
        """Run a specific analyzer."""
        print(f"\n" + "="*60)
        print(f"Running {analyzer_name.replace('_', ' ').title()} Analysis")
        print("="*60)
        
        start_time = time.time()
        
        try:
            if analyzer_name in self.analyzers:
                analyzer = self.analyzers[analyzer_name]
                if hasattr(analyzer, 'analyze_directory'):
                    result = analyzer.analyze_directory(str(self.project_root))
                elif hasattr(analyzer, 'analyze_project'):
                    result = analyzer.analyze_project()
                else:
                    result = {"error": f"No suitable analysis method found for {analyzer_name}"}
            else:
                result = {"error": f"Analyzer {analyzer_name} not available"}
            
            result["execution_time"] = time.time() - start_time
            
            # Save individual report
            report_path = self.reports_dir / f"{analyzer_name}_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(result, f, indent=2)
            
            return result
        except Exception as e:
            error_result = {
                "error": str(e),
                "execution_time": time.time() - start_time
            }
            
            # Save error report
            report_path = self.reports_dir / f"{analyzer_name}_error_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(error_result, f, indent=2)
            
            return error_result

    def run_all(self) -> dict[str, Any]:
        """Run the simplified code quality analysis pipeline."""
        print(f"\n{'='*80}")
        print("SIMPLIFIED CODE QUALITY ANALYSIS PIPELINE")
        print(f"{'='*80}")
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Total Python files: {len(self.file_paths)}")

        total_start = time.time()

        # Basic fixes
        self.results["fixes"] = {
            "syntax_fixes": self.run_syntax_fixes(),
            "import_fixes": self.run_import_fixes(),
            "async_fixes": self.run_async_fixes(),
            "type_hints": self.run_type_hints(),
        }

        # Basic validation
        self.results["validation"] = {
            "syntax_validation": self.run_syntax_validation(),
            "import_validation": self.run_import_validation(),
        }

        # Core analysis
        core_analyzers = [
            "syntax_validator", "linter", "undefined_names", "type_checker",
            "static_analysis", "ast_analysis", "metrics", "code_smell"
        ]
        
        self.results["core_analysis"] = {}
        for analyzer_name in core_analyzers:
            self.results["core_analysis"][analyzer_name] = self.run_analysis(analyzer_name)

        # Advanced analysis
        advanced_analyzers = [
            "architecture", "call_graph", "complexity", "dependency",
            "data_flow", "enhanced_dead_code", "performance", "concurrency",
            "error_handling", "code_duplication", "improved_signature"
        ]
        
        self.results["advanced_analysis"] = {}
        for analyzer_name in advanced_analyzers:
            self.results["advanced_analysis"][analyzer_name] = self.run_analysis(analyzer_name)

        # Quality analysis
        quality_analyzers = [
            "configuration", "documentation", "test_coverage"
        ]
        
        self.results["quality_analysis"] = {}
        for analyzer_name in quality_analyzers:
            self.results["quality_analysis"][analyzer_name] = self.run_analysis(analyzer_name)

        # Generate summary
        self.results["summary"] = self._generate_summary(time.time() - total_start)

        # Save main report
        report_path = self.reports_dir / f"simplified_pipeline_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Total execution time: {time.time() - total_start:.2f} seconds")
        print(f"Reports saved to: {self.reports_dir}")
        print(f"Main report: {report_path}")

        # Print summary
        self._print_summary()

        return self.results

    def _generate_summary(self, total_time: float) -> dict[str, Any]:
        """Generate summary of all results."""
        summary = {
            "timestamp": self.timestamp,
            "project_root": str(self.project_root),
            "total_execution_time": total_time,
            "total_files": len(self.file_paths),
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
                            "status": result.get("status", "unknown"),
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
        print(f"Total files analyzed: {summary['total_files']}")

        for category, tools in summary["categories"].items():
            print(f"\n{category.upper()}:")
            for tool, info in tools.items():
                print(f"  {tool}:")
                print(f"    Execution time: {info['execution_time']:.2f}s")
                print(f"    Status: {info['status']}")
                if info["issues_fixed"]:
                    print(f"    Issues fixed: {info['issues_fixed']}")
                if info["issues_found"]:
                    print(f"    Issues found: {info['issues_found']}")
                if info["files_processed"]:
                    print(f"    Files processed: {info['files_processed']}")


def main():
    """Main entry point for the simplified code quality analysis pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Simplified Code Quality Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simplified analysis
  python pipelines/pipeline_simplified.py
  
  # Run on specific project directory
  python pipelines/pipeline_simplified.py --project-root /path/to/project
        """
    )
    parser.add_argument("--project-root", default="/workspace/src",
                        help="Project root directory")

    args = parser.parse_args()

    pipeline = SimplifiedPipeline(project_root=args.project_root)

    # Run analysis
    pipeline.run_all()


if __name__ == "__main__":
    main()