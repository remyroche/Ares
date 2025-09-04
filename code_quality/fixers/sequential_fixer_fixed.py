"""
Sequential Auto-Fix Pipeline - Fixed Version with Dependency Management

Runs syntax fixing, linter analysis, and AST/compilation checking in sequence
with improved dependency handling and error management.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import dependency manager for safe imports
from utils.dependency_manager import dependency_manager, safe_import

# Safe imports with fallbacks
ImportAnalyzer, _ = safe_import("analyzers.import_analyzer", None)
SignatureAnalyzer, _ = safe_import("analyzers.improved_signature_analyzer", None)
LinterAnalyzer, _ = safe_import("analyzers.linter_analyzer", None)
SyntaxValidator, _ = safe_import("analyzers.syntax_validator", None)
AutoFixer, _ = safe_import("fixers.auto_fixer", None)

# Core imports with fallbacks
try:
    from core.config import CodeQualityConfig, get_default_config
except ImportError:
    CodeQualityConfig = None
    get_default_config = None

try:
    from utils.file_utils import find_python_files
except ImportError:
    find_python_files = None

# Import base pipeline for common functionality
from pipelines.base_pipeline import BasePipeline


class SequentialFixer(BasePipeline):
    """
    Sequential auto-fix pipeline that runs multiple quality tools in sequence.

    Pipeline:
    1. Auto-fix syntax and style issues
    2. Run linter analysis and error reporting
    3. Validate AST parsing and compilation
    4. Analyze imports for conflicts and circular dependencies
    5. Analyze function signatures for compatibility issues
    6. Generate comprehensive report
    """

    def __init__(self, config: CodeQualityConfig | None = None):
        # Initialize with default project root
        super().__init__("/workspace/src")
        
        # Get config with fallback
        if config is None:
            if get_default_config:
                self.config = get_default_config()
            else:
                # Create minimal config if get_default_config is not available
                self.config = self._create_fallback_config()
        else:
            self.config = config
        
        # Make auto-fix more conservative - remove aggressive tools
        if hasattr(self.config, 'auto_fix') and self.config.auto_fix:
            # Use a balanced set of tools - remove the most aggressive ones
            safe_tools = [
                "isort",      # Import sorting - very safe
                "autoflake",  # Remove unused imports/variables - mostly safe
                "pyupgrade",  # Upgrade Python syntax - fairly safe
                "yesqa",      # Remove unnecessary noqa comments - safe
                # Removed: black (can be aggressive with formatting)
                # Removed: yapf (another aggressive formatter)
                # Removed: autopep8 (can make unwanted changes)
                # Removed: docformatter (can break docstrings)
                # Removed: flynt (f-string conversion can be risky)
                # Removed: unify (quote changes can be problematic)
            ]
            self.config.auto_fix.tools = safe_tools
            self.config.auto_fix.aggressive = False
            self.config.auto_fix.max_line_length = 120

        # Print dependency status
        dependency_manager.print_dependency_status()

    def _create_fallback_config(self):
        """Create a fallback configuration when core config is not available."""
        class FallbackConfig:
            def __init__(self):
                self.auto_fix = FallbackAutoFix()
                self.analysis = FallbackAnalysis()
        
        class FallbackAutoFix:
            def __init__(self):
                self.enabled = True
                self.tools = ["isort", "autoflake"]
                self.aggressive = False
                self.max_line_length = 120
        
        class FallbackAnalysis:
            def __init__(self):
                self.linters = ["flake8"]
                self.exclude_patterns = ["__pycache__", "*.pyc", ".git", "venv", "env"]
        
        return FallbackConfig()

    def run_pipeline(self, target: str | list[str],
                    output_dir: str | None = None,
                    create_backups: bool = True,
                    run_pre_commit: bool = False) -> dict[str, Any]:
        """
        Run the complete sequential fixing pipeline.

        Args:
            target: Path to file/directory or list of paths
            output_dir: Directory to save reports
            create_backups: Whether to create backups before fixing

        Returns:
            Complete pipeline results
        """
        self._setup_execution_tracking()

        # Normalize target to list of files
        if isinstance(target, str):
            if os.path.isfile(target):
                target_files = [target]
                target_type = "file"
            else:
                if find_python_files:
                    target_files = find_python_files(target, self.config.analysis.exclude_patterns)
                else:
                    # Fallback: find Python files manually
                    target_files = self._find_python_files_fallback(target)
                target_type = "directory"
        else:
            target_files = target
            target_type = "file_list"

        self._print_pipeline_header("SEQUENTIAL AUTO-FIX PIPELINE")
        print(f"Target type: {target_type}")
        print(f"Target: {target if isinstance(target, str) else 'Multiple files'}")
        print(f"Files to process: {len(target_files)}")
        print(f"Backups enabled: {create_backups}")
        print(f"Output directory: {output_dir or 'None'}")
        print(f"Pre-commit: {'enabled' if run_pre_commit else 'disabled'}")
        print(f"Timestamp: {self.timestamp}")

        # Initialize results
        self.results = {
            "pipeline_info": {
                "target": target if isinstance(target, str) else str(target),
                "target_type": target_type,
                "total_files": len(target_files),
                "start_time": datetime.now().isoformat(),
                "timestamp": self.timestamp,
                "config_used": {
                    "auto_fix_tools": getattr(self.config.auto_fix, 'tools', []),
                    "linters": getattr(self.config.analysis, 'linters', []),
                    "max_line_length": getattr(self.config.auto_fix, 'max_line_length', 120),
                },
            },
            "step_results": {},
            "summary": {},
        }

        try:
            # Step 1: Auto-fix syntax and style issues
            self._print_section_header("STEP 1: AUTO-FIXING SYNTAX AND STYLE")
            fix_results = self._run_auto_fix(target_files, create_backups)
            self.results["step_results"]["auto_fix"] = fix_results

            # Step 2: Linter analysis and error reporting
            self._print_section_header("STEP 2: LINTER ANALYSIS AND ERROR REPORTING")
            linter_results = self._run_linter_analysis(target_files)
            self.results["step_results"]["linter_analysis"] = linter_results

            # Step 3: AST parsing and compilation validation
            self._print_section_header("STEP 3: AST PARSING AND COMPILATION VALIDATION")
            syntax_results = self._run_syntax_validation(target_files)
            self.results["step_results"]["syntax_validation"] = syntax_results

            # Step 4: Import analysis for conflicts and circular dependencies
            self._print_section_header("STEP 4: IMPORT ANALYSIS - CONFLICTS & CIRCULAR DEPENDENCIES")
            import_results = self._run_import_analysis(target_files)
            self.results["step_results"]["import_analysis"] = import_results

            # Step 5: Function signature analysis for compatibility
            self._print_section_header("STEP 5: FUNCTION SIGNATURE ANALYSIS - COMPATIBILITY CHECK")
            signature_results = self._run_signature_analysis(target_files)
            self.results["step_results"]["signature_analysis"] = signature_results

            # Optional: Pre-commit integration
            if run_pre_commit:
                self._print_section_header("OPTIONAL STEP: PRE-COMMIT INTEGRATION")
                pre_commit_results = self._run_pre_commit(target_files)
                self.results["step_results"]["pre_commit"] = pre_commit_results

            # Step 6: Generate comprehensive summary
            self._print_section_header("STEP 6: GENERATING COMPREHENSIVE SUMMARY")
            summary = self._generate_comprehensive_summary()
            self.results["summary"] = summary

            # Step 7: Save reports if requested
            if output_dir:
                self._save_reports(output_dir)

            # Step 8: Print final summary
            self._print_final_summary()

        except Exception as e:
            print(f"\nERROR: Pipeline failed: {e}")
            self.results["error"] = str(e)
            raise

        finally:
            total_time = self._finalize_execution_tracking()
            self.results["pipeline_info"]["end_time"] = datetime.now().isoformat()
            self.results["pipeline_info"]["duration"] = total_time

        return self.results

    def _find_python_files_fallback(self, target_dir: str) -> list[str]:
        """Fallback method to find Python files when find_python_files is not available."""
        python_files = []
        try:
            for root, dirs, files in os.walk(target_dir):
                # Skip certain directories
                dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'venv', 'env']]
                
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
        except Exception as e:
            print(f"Error finding Python files: {e}")
        
        return python_files

    def _run_auto_fix(self, files: list[str], create_backups: bool) -> dict[str, Any]:
        """Run auto-fixing on the target files."""
        if not AutoFixer:
            return self._handle_error(
                Exception("AutoFixer not available - missing dependencies"),
                "auto_fix"
            )

        if not getattr(self.config.auto_fix, 'enabled', True):
            print("Auto-fixing is disabled in configuration.")
            return {"status": "disabled", "reason": "Auto-fixing disabled in config"}

        print(f"Running auto-fix on {len(files)} files...")

        # Group files by directory for efficient processing
        files_by_dir = {}
        for file_path in files:
            dir_path = str(Path(file_path).parent)
            if dir_path not in files_by_dir:
                files_by_dir[dir_path] = []
            files_by_dir[dir_path].append(file_path)

        all_fix_results = {}
        total_files_processed = 0

        for dir_path, dir_files in files_by_dir.items():
            print(f"Processing directory: {dir_path}")

            try:
                fixer = AutoFixer(self.config)
                if len(dir_files) == 1:
                    # Single file
                    fix_results = fixer.fix_file(dir_files[0])
                else:
                    # Multiple files in directory
                    fix_results = fixer.fix_all(dir_path)

                all_fix_results[dir_path] = fix_results
                total_files_processed += len(dir_files)

                # Print summary for this directory
                summary = fixer.get_fix_summary()
                print(f"  Tools run: {', '.join(summary['tools_run'])}")
                print(f"  Successful: {', '.join(summary['successful_tools'])}")
                if summary["failed_tools"]:
                    print(f"  Failed: {', '.join(summary['failed_tools'])}")

            except Exception as e:
                print(f"  Error processing {dir_path}: {e}")
                all_fix_results[dir_path] = {"status": "error", "error": str(e)}

        # Aggregate results
        overall_status = "success"
        failed_tools = set()
        successful_tools = set()

        for dir_results in all_fix_results.values():
            if isinstance(dir_results, dict):
                for tool, result in dir_results.items():
                    if isinstance(result, dict):
                        if result.get("status") == "success":
                            successful_tools.add(tool)
                        elif result.get("status") in ["failed", "error"]:
                            failed_tools.add(tool)
                            overall_status = "partial"

        if failed_tools:
            overall_status = "partial"

        return {
            "status": overall_status,
            "total_files_processed": total_files_processed,
            "successful_tools": list(successful_tools),
            "failed_tools": list(failed_tools),
            "directory_results": all_fix_results,
        }

    def _run_linter_analysis(self, files: list[str]) -> dict[str, Any]:
        """Run linter analysis on the target files."""
        if not LinterAnalyzer:
            return self._handle_error(
                Exception("LinterAnalyzer not available - missing dependencies"),
                "linter_analysis"
            )

        print(f"Running linter analysis on {len(files)} files...")

        try:
            # Find the common parent directory
            if len(files) == 1:
                target_dir = str(Path(files[0]).parent)
            else:
                # Find common ancestor directory
                paths = [Path(f) for f in files]
                common_prefix = Path(os.path.commonpath([str(p) for p in paths]))
                target_dir = str(common_prefix)

            linter_analyzer = LinterAnalyzer(self.config)
            linter_results = linter_analyzer.analyze_directory(target_dir)

            # Filter results to only include our target files
            filtered_results = {
                "total_issues": 0,
                "total_files_with_issues": 0,
                "total_errors": 0,
                "total_warnings": 0,
                "by_file": {},
                "by_directory": linter_results.get("by_directory", {}),
                "by_linter": linter_results.get("by_linter", {}),
                "by_error_type": linter_results.get("by_error_type", {}),
            }

            for file_path in files:
                if file_path in linter_results.get("by_file", {}):
                    file_issues = linter_results["by_file"][file_path]
                    filtered_results["by_file"][file_path] = file_issues
                    filtered_results["total_issues"] += len(file_issues)
                    filtered_results["total_files_with_issues"] += 1

                    for issue in file_issues:
                        if issue.get("severity") == "error":
                            filtered_results["total_errors"] += 1
                        else:
                            filtered_results["total_warnings"] += 1

            return {
                "status": "success",
                "results": filtered_results,
                "full_results": linter_results,
            }

        except Exception as e:
            return self._handle_error(e, "linter_analysis")

    def _run_pre_commit(self, files: list[str]) -> dict[str, Any]:
        """Run pre-commit hooks across repository or scoped directory."""
        try:
            import subprocess

            if len(files) == 1:
                target_dir = str(Path(files[0]).parent)
            else:
                paths = [Path(f) for f in files]
                target_dir = str(Path(os.path.commonpath([str(p) for p in paths])))

            cmd = [sys.executable, "-m", "pre_commit", "run", "--all-files"]
            result = subprocess.run(cmd, cwd=target_dir, check=False, capture_output=True, text=True)

            return {
                "status": "success" if result.returncode in (0, 1) else "failed",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }
        except Exception as e:
            return self._handle_error(e, "pre_commit")

    def _run_syntax_validation(self, files: list[str]) -> dict[str, Any]:
        """Run syntax validation on the target files."""
        if not SyntaxValidator:
            return self._handle_error(
                Exception("SyntaxValidator not available - missing dependencies"),
                "syntax_validation"
            )

        print(f"Running syntax validation on {len(files)} files...")

        try:
            # Find the common parent directory
            if len(files) == 1:
                target_dir = str(Path(files[0]).parent)
            else:
                # Find common ancestor directory
                paths = [Path(f) for f in files]
                common_prefix = Path(os.path.commonpath([str(p) for p in paths]))
                target_dir = str(common_prefix)

            syntax_validator = SyntaxValidator(self.config)
            syntax_results = syntax_validator.validate_directory(target_dir)

            # Filter results to only include our target files
            filtered_results = {
                "summary": {
                    "total_files": len(files),
                    "valid_files": 0,
                    "invalid_files": 0,
                    "ast_parseable_files": 0,
                    "compilable_files": 0,
                    "total_errors": 0,
                    "total_ast_nodes": 0,
                },
                "errors_by_file": {},
                "file_details": {},
            }

            for file_path in files:
                if file_path in syntax_results.get("file_details", {}):
                    file_details = syntax_results["file_details"][file_path]
                    filtered_results["file_details"][file_path] = file_details

                    if file_details.get("syntax_valid", False):
                        filtered_results["summary"]["valid_files"] += 1
                    else:
                        filtered_results["summary"]["invalid_files"] += 1

                    if file_details.get("ast_parseable", False):
                        filtered_results["summary"]["ast_parseable_files"] += 1

                    if file_details.get("compilable", False):
                        filtered_results["summary"]["compilable_files"] += 1

                    # Count errors for this file
                    file_errors = syntax_results.get("errors_by_file", {}).get(file_path, [])
                    filtered_results["errors_by_file"][file_path] = file_errors
                    filtered_results["summary"]["total_errors"] += len(file_errors)

            # Get total AST nodes
            filtered_results["summary"]["total_ast_nodes"] = syntax_results.get("summary", {}).get("total_ast_nodes", 0)

            return {
                "status": "success",
                "results": filtered_results,
                "full_results": syntax_results,
            }

        except Exception as e:
            return self._handle_error(e, "syntax_validation")

    def _run_import_analysis(self, files: list[str]) -> dict[str, Any]:
        """Run import analysis on the target files."""
        if not ImportAnalyzer:
            return self._handle_error(
                Exception("ImportAnalyzer not available - missing dependencies"),
                "import_analysis"
            )

        print(f"Running import analysis on {len(files)} files...")

        try:
            # Find the common parent directory
            if len(files) == 1:
                target_dir = str(Path(files[0]).parent)
            else:
                # Find common ancestor directory
                paths = [Path(f) for f in files]
                common_prefix = Path(os.path.commonpath([str(p) for p in paths]))
                target_dir = str(common_prefix)

            import_analyzer = ImportAnalyzer(self.config)
            import_results = import_analyzer.analyze_directory(target_dir)

            # Filter results to only include our target files
            filtered_results = {
                "summary": {
                    "total_files_analyzed": len(files),
                    "total_imports": 0,
                    "total_issues": 0,
                    "duplicate_imports": 0,
                    "circular_dependencies": 0,
                    "conflicting_imports": 0,
                },
                "issues": {
                    "duplicate_imports": [],
                    "circular_dependencies": [],
                    "conflicting_imports": [],
                },
            }

            # Filter issues to only include our target files
            for issue_type in ["duplicate_imports", "circular_dependencies", "conflicting_imports"]:
                if issue_type in import_results.get("issues", {}):
                    for issue in import_results["issues"][issue_type]:
                        if issue.get("file") in files:
                            filtered_results["issues"][issue_type].append(issue)
                            filtered_results["summary"][f"{issue_type}"] += 1
                            filtered_results["summary"]["total_issues"] += 1

            # Get total imports for our files
            for file_path in files:
                if file_path in import_results.get("files", {}):
                    filtered_results["summary"]["total_imports"] += import_results["files"][file_path].get("total_imports", 0)

            return {
                "status": "success",
                "results": filtered_results,
                "full_results": import_results,
            }

        except Exception as e:
            return self._handle_error(e, "import_analysis")

    def _run_signature_analysis(self, files: list[str]) -> dict[str, Any]:
        """Run function signature analysis on the target files."""
        if not SignatureAnalyzer:
            return self._handle_error(
                Exception("SignatureAnalyzer not available - missing dependencies"),
                "signature_analysis"
            )

        print(f"Running function signature analysis on {len(files)} files...")

        try:
            # Find the common parent directory
            if len(files) == 1:
                target_dir = str(Path(files[0]).parent)
            else:
                # Find common ancestor directory
                paths = [Path(f) for f in files]
                common_prefix = Path(os.path.commonpath([str(p) for p in paths]))
                target_dir = str(common_prefix)

            signature_analyzer = SignatureAnalyzer(self.config)
            signature_results = signature_analyzer.analyze_directory(target_dir)

            # Filter results to only include our target files
            filtered_results = {
                "summary": {
                    "total_files_analyzed": len(files),
                    "total_functions": 0,
                    "total_function_calls": 0,
                    "total_issues": 0,
                    "signature_changes": 0,
                    "compatibility_issues": 0,
                    "missing_functions": 0,
                    "unused_functions": 0,
                },
                "issues": {
                    "signature_changes": [],
                    "compatibility_issues": [],
                    "missing_functions": [],
                    "unused_functions": [],
                },
            }

            # Filter issues to only include our target files
            for issue_type in ["signature_changes", "compatibility_issues", "missing_functions", "unused_functions"]:
                if issue_type in signature_results.get("issues", {}):
                    for issue in signature_results["issues"][issue_type]:
                        if issue.get("file") in files:
                            filtered_results["issues"][issue_type].append(issue)
                            filtered_results["summary"][f"{issue_type}"] += 1
                            filtered_results["summary"]["total_issues"] += 1

            # Get function counts for our files
            for file_path in files:
                if file_path in signature_results.get("functions", {}):
                    filtered_results["summary"]["total_functions"] += len(signature_results["functions"][file_path])

                if file_path in signature_results.get("calls", {}):
                    filtered_results["summary"]["total_function_calls"] += len(signature_results["calls"][file_path])

            return {
                "status": "success",
                "results": filtered_results,
                "full_results": signature_results,
            }

        except Exception as e:
            return self._handle_error(e, "signature_analysis")

    def _generate_comprehensive_summary(self) -> dict[str, Any]:
        """Generate a comprehensive summary of all pipeline steps."""
        summary = {
            "overall_status": "success",
            "step_statuses": {},
            "metrics": {},
            "recommendations": [],
        }

        # Check step statuses
        for step_name, step_results in self.results["step_results"].items():
            status = step_results.get("status", "unknown")
            summary["step_statuses"][step_name] = status

            if status in ["error", "failed"]:
                summary["overall_status"] = "failed"
            elif status == "partial":
                if summary["overall_status"] == "success":
                    summary["overall_status"] = "partial"

        # Calculate metrics
        auto_fix = self.results["step_results"].get("auto_fix", {})
        linter = self.results["step_results"].get("linter_analysis", {})
        syntax = self.results["step_results"].get("syntax_validation", {})
        imports = self.results["step_results"].get("import_analysis", {})
        signatures = self.results["step_results"].get("signature_analysis", {})

        summary["metrics"] = {
            "files_processed": auto_fix.get("total_files_processed", 0),
            "auto_fix_successful_tools": len(auto_fix.get("successful_tools", [])),
            "auto_fix_failed_tools": len(auto_fix.get("failed_tools", [])),
            "linter_issues": linter.get("results", {}).get("total_issues", 0) if linter.get("status") == "success" else 0,
            "syntax_errors": syntax.get("results", {}).get("summary", {}).get("total_errors", 0) if syntax.get("status") == "success" else 0,
            "valid_files": syntax.get("results", {}).get("summary", {}).get("valid_files", 0) if syntax.get("status") == "success" else 0,
            "invalid_files": syntax.get("results", {}).get("summary", {}).get("invalid_files", 0) if syntax.get("status") == "success" else 0,
            "import_issues": imports.get("results", {}).get("summary", {}).get("total_issues", 0) if imports.get("status") == "success" else 0,
            "signature_issues": signatures.get("results", {}).get("summary", {}).get("total_issues", 0) if signatures.get("status") == "success" else 0,
            "pre_commit_return_code": self.results.get("step_results", {}).get("pre_commit", {}).get("return_code"),
        }

        # Generate recommendations
        if summary["metrics"]["syntax_errors"] > 0:
            summary["recommendations"].append({
                "priority": "high",
                "category": "syntax",
                "message": f"Fix {summary['metrics']['syntax_errors']} syntax errors to ensure code can run",
            })

        if summary["metrics"]["linter_issues"] > 0:
            summary["recommendations"].append({
                "priority": "medium",
                "category": "quality",
                "message": f"Address {summary['metrics']['linter_issues']} linting issues for better code quality",
            })

        if summary["metrics"]["import_issues"] > 0:
            summary["recommendations"].append({
                "priority": "high",
                "category": "imports",
                "message": f"Resolve {summary['metrics']['import_issues']} import conflicts and circular dependencies",
            })

        if summary["metrics"]["signature_issues"] > 0:
            summary["recommendations"].append({
                "priority": "high",
                "category": "compatibility",
                "message": f"Fix {summary['metrics']['signature_issues']} function signature compatibility issues",
            })

        if auto_fix.get("failed_tools"):
            summary["recommendations"].append({
                "priority": "medium",
                "category": "tools",
                "message": f"Some auto-fix tools failed: {', '.join(auto_fix['failed_tools'])}",
            })

        if summary["metrics"]["invalid_files"] > 0:
            summary["recommendations"].append({
                "priority": "high",
                "category": "syntax",
                "message": f"{summary['metrics']['invalid_files']} files have syntax errors that prevent execution",
            })

        return summary

    def _save_reports(self, output_dir: str) -> None:
        """Save pipeline reports to the specified output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\nSaving pipeline reports to: {output_path}")

        # Save main pipeline report with timestamp
        pipeline_file = output_path / f"sequential_fixer_pipeline_report_{self.timestamp}.json"
        with open(pipeline_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Pipeline report saved: {pipeline_file}")

        # Save individual step reports with timestamps
        for step_name, step_results in self.results["step_results"].items():
            if step_results.get("status") == "success":
                step_file = output_path / f"{step_name}_report_{self.timestamp}.json"
                with open(step_file, "w") as f:
                    json.dump(step_results, f, indent=2)
                print(f"{step_name} report saved: {step_file}")

        # Save compact HTML summary
        try:
            html_path = output_path / f"sequential_fixer_summary_{self.timestamp}.html"
            html = [
                "<html><head><meta charset='utf-8'><title>Sequential Fixer Summary</title></head><body>",
                f"<h1>Sequential Fixer Summary ({self.timestamp})</h1>",
                f"<p>Status: {self.results['summary'].get('overall_status','unknown')}</p>",
                "<h2>Metrics</h2><ul>",
            ]
            for k, v in self.results["summary"].get("metrics", {}).items():
                html.append(f"<li>{k}: {v}</li>")
            html.append("</ul><h2>Recommendations</h2><ol>")
            for rec in self.results["summary"].get("recommendations", []):
                html.append(f"<li>[{rec.get('priority','')}] {rec.get('message','')}</li>")
            html.append("</ol></body></html>")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write("\n".join(html))
            print(f"HTML summary saved: {html_path}")
        except Exception:
            pass

    def _print_final_summary(self) -> None:
        """Print the final pipeline summary."""
        summary = self.results["summary"]

        print("\n" + "="*70)
        print("SEQUENTIAL AUTO-FIX PIPELINE COMPLETED")
        print("="*70)
        print(f"Overall Status: {summary['overall_status'].upper()}")
        print(f"Timestamp: {self.timestamp}")

        print("\nStep Statuses:")
        for step, status in summary["step_statuses"].items():
            print(f"  {step.replace('_', ' ').title()}: {status.upper()}")

        print("\nMetrics:")
        metrics = summary["metrics"]
        print(f"  Files processed: {metrics['files_processed']}")
        print(f"  Auto-fix successful tools: {metrics['auto_fix_successful_tools']}")
        print(f"  Auto-fix failed tools: {metrics['auto_fix_failed_tools']}")
        print(f"  Linter issues: {metrics['linter_issues']}")
        print(f"  Syntax errors: {metrics['syntax_errors']}")
        print(f"  Valid files: {metrics['valid_files']}")
        print(f"  Invalid files: {metrics['invalid_files']}")
        print(f"  Import issues: {metrics['import_issues']}")
        print(f"  Signature issues: {metrics['signature_issues']}")

        if summary["recommendations"]:
            print("\nRecommendations:")
            for i, rec in enumerate(summary["recommendations"], 1):
                print(f"  {i}. [{rec['priority'].upper()}] {rec['message']}")

        # Check if duration exists before trying to print it
        if "duration" in self.results["pipeline_info"]:
            duration = self.results["pipeline_info"]["duration"]
            print(f"\nPipeline completed in {duration:.2f} seconds")
        elif self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            print(f"\nPipeline completed in {duration:.2f} seconds")


def main():
    """Command-line interface for the sequential fixer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Sequential Auto-Fix Pipeline - Fix syntax, run linters, validate AST/compilation",
    )
    parser.add_argument("--target", required=True,
                       help="Path to Python file, directory, or comma-separated list of files")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", help="Output directory for reports")
    parser.add_argument("--no-backups", action="store_true", help="Disable backup creation")
    parser.add_argument("--pre-commit", action="store_true", help="Run pre-commit hooks after fixes")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        if get_default_config:
            from core.config import load_config
            config = load_config(args.config)
        else:
            print("Warning: Configuration loading not available, using default config")
            config = None
    else:
        config = None

    # Parse target
    if "," in args.target:
        # Comma-separated list of files
        target = [f.strip() for f in args.target.split(",")]
    else:
        target = args.target

    # Run pipeline
    with SequentialFixer(config) as fixer:
        results = fixer.run_pipeline(
            target=target,
            output_dir=args.output,
            create_backups=not args.no_backups,
            run_pre_commit=args.pre_commit,
        )

        # Exit with appropriate code
        if results["summary"]["overall_status"] == "success":
            return 0
        if results["summary"]["overall_status"] == "partial":
            return 1
        return 2


if __name__ == "__main__":
    sys.exit(main())