"""
Production Linter Runner Plugin

A robust, production-ready plugin for running multiple linters on Python code with
comprehensive error handling, result aggregation, and detailed reporting.
"""

import subprocess
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, Any, Set, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from code_quality.plugins.base_plugin import DirectoryProcessorPlugin, PluginMetadata, PluginCategory, PluginPriority


class ProductionLinterPlugin(DirectoryProcessorPlugin):
    """
    Production-ready plugin for running multiple linters on Python code.
    
    Features:
    - Multiple linter support (flake8, pylint, mypy, black, isort)
    - Parallel linter execution
    - Comprehensive result aggregation
    - Configurable linter options
    - Detailed error reporting and metrics
    - Backup creation and rollback capabilities
    - Performance monitoring and optimization
    """
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="production_linter_runner",
            version="2.0.0",
            description="Production-ready multi-linter runner with comprehensive analysis and reporting",
            author="Code Quality Team",
            category=PluginCategory.LINTING,
            priority=PluginPriority.MEDIUM,
            dependencies=[],
            tags={"linting", "analysis", "python", "production", "multi-tool"},
            required_packages=[],
            optional_packages=["flake8", "pylint", "mypy", "black", "isort", "bandit"],
            configuration_schema={
                "create_backups": {"type": "boolean", "default": True},
                "backup_suffix": {"type": "string", "default": ".bak"},
                "linters": {"type": "list", "default": ["flake8", "pylint", "mypy"]},
                "parallel_execution": {"type": "boolean", "default": True},
                "max_workers": {"type": "integer", "default": 4},
                "timeout_per_linter": {"type": "integer", "default": 300},
                "max_line_length": {"type": "integer", "default": 120},
                "ignore_errors": {"type": "list", "default": []},
                "exclude_patterns": {"type": "list", "default": ["__pycache__", "*.pyc", "test_*.py"]},
                "output_format": {"type": "string", "default": "json"},
                "fix_issues": {"type": "boolean", "default": False},
                "aggressive_mode": {"type": "boolean", "default": False},
                "custom_configs": {"type": "dict", "default": {}},
                "report_coverage": {"type": "boolean", "default": True},
                "generate_reports": {"type": "boolean", "default": True}
            }
        )
    
    def is_available(self) -> bool:
        """Check if plugin is available."""
        # Check if at least one linter is available
        available_linters = self._get_available_linters()
        return len(available_linters) > 0
    
    def get_supported_file_types(self) -> Set[str]:
        """Get supported file types."""
        return {'.py', '.pyi', '.pyw'}
    
    def process_directory(self, directory_path: Path, context) -> Dict[str, Any]:
        """
        Process a directory with multiple linters.
        
        Args:
            directory_path: Path to the directory to process
            context: Plugin execution context
            
        Returns:
            Dict[str, Any]: Comprehensive processing result
        """
        result = {
            "success": True,
            "files_processed": 0,
            "files_fixed": 0,
            "files_failed": 0,
            "issues_found": 0,
            "issues_fixed": 0,
            "errors": [],
            "warnings": [],
            "output_data": {},
            "backup_created": False,
            "processing_time": 0.0,
            "linter_results": {},
            "performance_metrics": {}
        }
        
        start_time = datetime.now()
        
        try:
            # Validate directory
            if not self._validate_directory(directory_path):
                result["success"] = False
                result["errors"].append("Directory validation failed")
                return result
            
            # Create backup if configured
            backup_path = None
            if self.configuration.get("create_backups", True) and not context.dry_run:
                backup_path = self._create_backup(directory_path)
                if backup_path:
                    result["backup_created"] = True
                    result["backup_path"] = str(backup_path)
            
            # Get available linters
            available_linters = self._get_available_linters()
            if not available_linters:
                result["success"] = False
                result["errors"].append("No linters available")
                return result
            
            # Find Python files
            python_files = self._find_python_files(directory_path)
            result["files_processed"] = len(python_files)
            
            if not python_files:
                result["warnings"].append("No Python files found to process")
                return result
            
            # Run linters
            linter_results = {}
            total_issues = 0
            
            if self.configuration.get("parallel_execution", True):
                linter_results = self._run_linters_parallel(
                    available_linters, python_files, context
                )
            else:
                linter_results = self._run_linters_sequential(
                    available_linters, python_files, context
                )
            
            # Aggregate results
            for linter_name, linter_result in linter_results.items():
                if linter_result.get("success", False):
                    total_issues += linter_result.get("issues_found", 0)
                else:
                    result["errors"].extend(linter_result.get("errors", []))
                    result["warnings"].extend(linter_result.get("warnings", []))
            
            result["issues_found"] = total_issues
            result["linter_results"] = linter_results
            result["output_data"] = {
                "available_linters": available_linters,
                "linter_summary": self._generate_linter_summary(linter_results),
                "performance_metrics": self._calculate_performance_metrics(linter_results)
            }
            
            # Generate reports if configured
            if self.configuration.get("generate_reports", True) and not context.dry_run:
                report_path = self._generate_linter_report(linter_results, directory_path)
                if report_path:
                    result["output_data"]["report_path"] = str(report_path)
            
            # Note: Linters typically don't fix issues, they just report them
            result["issues_fixed"] = 0
        
        except Exception as e:
            result["success"] = False
            result["errors"].append(f"Unexpected error: {str(e)}")
            result["warnings"].append(f"Exception during processing: {type(e).__name__}")
        
        finally:
            result["processing_time"] = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def _validate_directory(self, directory_path: Path) -> bool:
        """Validate that the directory can be processed."""
        try:
            if not directory_path.exists():
                return False
            
            if not directory_path.is_dir():
                return False
            
            return True
        except Exception:
            return False
    
    def _create_backup(self, directory_path: Path) -> Optional[Path]:
        """Create a backup of the directory."""
        try:
            backup_suffix = self.configuration.get("backup_suffix", ".bak")
            backup_path = directory_path.with_suffix(directory_path.suffix + backup_suffix)
            
            # Ensure backup path is unique
            counter = 1
            while backup_path.exists():
                backup_path = directory_path.with_suffix(f"{directory_path.suffix}.{counter}{backup_suffix}")
                counter += 1
            
            shutil.copytree(directory_path, backup_path)
            return backup_path
        except Exception:
            return None
    
    def _get_available_linters(self) -> List[str]:
        """Get list of available linters."""
        available = []
        configured_linters = self.configuration.get("linters", ["flake8", "pylint", "mypy"])
        
        for linter in configured_linters:
            try:
                subprocess.run([linter, "--version"], 
                             capture_output=True, check=True, timeout=5)
                available.append(linter)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        return available
    
    def _find_python_files(self, directory_path: Path) -> List[Path]:
        """Find Python files in directory."""
        python_files = []
        exclude_patterns = self.configuration.get("exclude_patterns", ["__pycache__", "*.pyc", "test_*.py"])
        
        for py_file in directory_path.rglob("*.py"):
            # Check if file should be excluded
            should_exclude = False
            for pattern in exclude_patterns:
                if pattern in str(py_file):
                    should_exclude = True
                    break
            
            if not should_exclude:
                python_files.append(py_file)
        
        return python_files
    
    def _run_linters_parallel(self, linters: List[str], files: List[Path], context) -> Dict[str, Dict[str, Any]]:
        """Run linters in parallel."""
        results = {}
        max_workers = self.configuration.get("max_workers", 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all linter tasks
            future_to_linter = {
                executor.submit(self._run_single_linter, linter, files, context): linter
                for linter in linters
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_linter):
                linter_name = future_to_linter[future]
                try:
                    result = future.result()
                    results[linter_name] = result
                except Exception as e:
                    results[linter_name] = {
                        "success": False,
                        "issues_found": 0,
                        "output": "",
                        "errors": [f"Linter execution failed: {e}"],
                        "execution_time": 0.0
                    }
        
        return results
    
    def _run_linters_sequential(self, linters: List[str], files: List[Path], context) -> Dict[str, Dict[str, Any]]:
        """Run linters sequentially."""
        results = {}
        
        for linter in linters:
            try:
                result = self._run_single_linter(linter, files, context)
                results[linter] = result
            except Exception as e:
                results[linter] = {
                    "success": False,
                    "issues_found": 0,
                    "output": "",
                    "errors": [f"Linter execution failed: {e}"],
                    "execution_time": 0.0
                }
        
        return results
    
    def _run_single_linter(self, linter: str, files: List[Path], context) -> Dict[str, Any]:
        """
        Run a single linter on files.
        
        Args:
            linter: Name of the linter to run
            files: List of files to lint
            context: Plugin execution context
            
        Returns:
            Dict[str, Any]: Linter result
        """
        result = {
            "linter": linter,
            "success": True,
            "issues_found": 0,
            "output": "",
            "errors": [],
            "warnings": [],
            "execution_time": 0.0,
            "return_code": 0
        }
        
        start_time = datetime.now()
        
        try:
            # Build command based on linter
            cmd = self._build_linter_command(linter, files)
            
            # Run the linter
            if not context.dry_run:
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.configuration.get("timeout_per_linter", 300),
                    cwd=str(context.project_root)
                )
                
                result["output"] = process.stdout + process.stderr
                result["return_code"] = process.returncode
                
                # Count issues (enhanced parsing)
                result["issues_found"] = self._count_issues_enhanced(result["output"], linter)
                
                # Check for success (some linters return non-zero for issues found)
                if process.returncode not in [0, 1]:  # 0 = success, 1 = issues found
                    result["success"] = False
                    result["errors"].append(f"Linter {linter} failed with return code {process.returncode}")
            else:
                result["output"] = f"Dry run: Would execute {' '.join(cmd)}"
                result["issues_found"] = 0
        
        except subprocess.TimeoutExpired:
            result["success"] = False
            result["errors"].append(f"Linter {linter} timed out after {self.configuration.get('timeout_per_linter', 300)} seconds")
        except Exception as e:
            result["success"] = False
            result["errors"].append(f"Error running {linter}: {e}")
        
        finally:
            result["execution_time"] = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def _build_linter_command(self, linter: str, files: List[Path]) -> List[str]:
        """Build command for a specific linter."""
        if linter == "flake8":
            return self._build_flake8_command(files)
        elif linter == "pylint":
            return self._build_pylint_command(files)
        elif linter == "mypy":
            return self._build_mypy_command(files)
        elif linter == "black":
            return self._build_black_command(files)
        elif linter == "isort":
            return self._build_isort_command(files)
        elif linter == "bandit":
            return self._build_bandit_command(files)
        else:
            return [linter] + [str(f) for f in files]
    
    def _build_flake8_command(self, files: List[Path]) -> List[str]:
        """Build flake8 command."""
        cmd = ["flake8"]
        
        # Add configuration options
        max_line_length = self.configuration.get("max_line_length", 120)
        cmd.extend(["--max-line-length", str(max_line_length)])
        
        # Add ignore patterns
        ignore_errors = self.configuration.get("ignore_errors", [])
        if ignore_errors:
            cmd.extend(["--ignore", ",".join(ignore_errors)])
        
        # Add output format
        output_format = self.configuration.get("output_format", "json")
        if output_format == "json":
            cmd.extend(["--format", "json"])
        
        # Add custom configs
        custom_configs = self.configuration.get("custom_configs", {})
        if "flake8" in custom_configs:
            for key, value in custom_configs["flake8"].items():
                cmd.extend([f"--{key}", str(value)])
        
        # Add files
        cmd.extend([str(f) for f in files])
        
        return cmd
    
    def _build_pylint_command(self, files: List[Path]) -> List[str]:
        """Build pylint command."""
        cmd = ["pylint"]
        
        # Add configuration options
        max_line_length = self.configuration.get("max_line_length", 120)
        cmd.extend(["--max-line-length", str(max_line_length)])
        
        # Add output format
        output_format = self.configuration.get("output_format", "json")
        if output_format == "json":
            cmd.extend(["--output-format", "json"])
        
        # Add custom configs
        custom_configs = self.configuration.get("custom_configs", {})
        if "pylint" in custom_configs:
            for key, value in custom_configs["pylint"].items():
                cmd.extend([f"--{key}", str(value)])
        
        # Add files
        cmd.extend([str(f) for f in files])
        
        return cmd
    
    def _build_mypy_command(self, files: List[Path]) -> List[str]:
        """Build mypy command."""
        cmd = ["mypy"]
        
        # Add output format
        output_format = self.configuration.get("output_format", "json")
        if output_format == "json":
            cmd.extend(["--json-report", "/tmp/mypy-report.json"])
        
        # Add custom configs
        custom_configs = self.configuration.get("custom_configs", {})
        if "mypy" in custom_configs:
            for key, value in custom_configs["mypy"].items():
                cmd.extend([f"--{key}", str(value)])
        
        # Add files
        cmd.extend([str(f) for f in files])
        
        return cmd
    
    def _build_black_command(self, files: List[Path]) -> List[str]:
        """Build black command."""
        cmd = ["black"]
        
        # Add configuration options
        max_line_length = self.configuration.get("max_line_length", 120)
        cmd.extend(["--line-length", str(max_line_length)])
        
        # Add check mode if not fixing
        if not self.configuration.get("fix_issues", False):
            cmd.append("--check")
        
        # Add custom configs
        custom_configs = self.configuration.get("custom_configs", {})
        if "black" in custom_configs:
            for key, value in custom_configs["black"].items():
                cmd.extend([f"--{key}", str(value)])
        
        # Add files
        cmd.extend([str(f) for f in files])
        
        return cmd
    
    def _build_isort_command(self, files: List[Path]) -> List[str]:
        """Build isort command."""
        cmd = ["isort"]
        
        # Add configuration options
        max_line_length = self.configuration.get("max_line_length", 120)
        cmd.extend(["--line-length", str(max_line_length)])
        
        # Add check mode if not fixing
        if not self.configuration.get("fix_issues", False):
            cmd.append("--check-only")
        
        # Add custom configs
        custom_configs = self.configuration.get("custom_configs", {})
        if "isort" in custom_configs:
            for key, value in custom_configs["isort"].items():
                cmd.extend([f"--{key}", str(value)])
        
        # Add files
        cmd.extend([str(f) for f in files])
        
        return cmd
    
    def _build_bandit_command(self, files: List[Path]) -> List[str]:
        """Build bandit command."""
        cmd = ["bandit"]
        
        # Add output format
        output_format = self.configuration.get("output_format", "json")
        if output_format == "json":
            cmd.extend(["-f", "json"])
        
        # Add custom configs
        custom_configs = self.configuration.get("custom_configs", {})
        if "bandit" in custom_configs:
            for key, value in custom_configs["bandit"].items():
                cmd.extend([f"-{key}", str(value)])
        
        # Add files
        cmd.extend([str(f) for f in files])
        
        return cmd
    
    def _count_issues_enhanced(self, output: str, linter: str) -> int:
        """Enhanced issue counting for different linters."""
        if not output:
            return 0
        
        try:
            # Try to parse JSON output first
            if output.strip().startswith('[') or output.strip().startswith('{'):
                data = json.loads(output)
                if isinstance(data, list):
                    return len(data)
                elif isinstance(data, dict) and "results" in data:
                    return len(data["results"])
        except json.JSONDecodeError:
            pass
        
        # Fallback to line-based counting
        lines = output.strip().split('\n')
        issue_count = 0
        
        for line in lines:
            if linter == "flake8":
                # Flake8 format: filename:line:column: code message
                if ':' in line and any(char.isdigit() for char in line):
                    issue_count += 1
            elif linter == "pylint":
                # Pylint format: filename:line:column: message
                if ':' in line and any(char.isdigit() for char in line):
                    issue_count += 1
            elif linter == "mypy":
                # MyPy format: filename:line: error: message
                if 'error:' in line:
                    issue_count += 1
            elif linter == "black":
                # Black format: would reformat filename
                if 'would reformat' in line:
                    issue_count += 1
            elif linter == "isort":
                # isort format: ERROR: filename Imports are incorrectly sorted
                if 'ERROR:' in line and 'incorrectly sorted' in line:
                    issue_count += 1
            elif linter == "bandit":
                # Bandit format: Issue: [severity] message
                if 'Issue:' in line:
                    issue_count += 1
        
        return issue_count
    
    def _generate_linter_summary(self, linter_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of linter results."""
        summary = {
            "total_linters": len(linter_results),
            "successful_linters": 0,
            "failed_linters": 0,
            "total_issues": 0,
            "total_execution_time": 0.0,
            "linter_breakdown": {}
        }
        
        for linter_name, result in linter_results.items():
            if result.get("success", False):
                summary["successful_linters"] += 1
            else:
                summary["failed_linters"] += 1
            
            issues = result.get("issues_found", 0)
            execution_time = result.get("execution_time", 0.0)
            
            summary["total_issues"] += issues
            summary["total_execution_time"] += execution_time
            
            summary["linter_breakdown"][linter_name] = {
                "issues_found": issues,
                "execution_time": execution_time,
                "success": result.get("success", False)
            }
        
        return summary
    
    def _calculate_performance_metrics(self, linter_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics from linter results."""
        metrics = {
            "average_execution_time": 0.0,
            "fastest_linter": None,
            "slowest_linter": None,
            "total_files_processed": 0,
            "issues_per_second": 0.0
        }
        
        if not linter_results:
            return metrics
        
        execution_times = []
        total_issues = 0
        
        for linter_name, result in linter_results.items():
            execution_time = result.get("execution_time", 0.0)
            issues = result.get("issues_found", 0)
            
            execution_times.append((linter_name, execution_time))
            total_issues += issues
        
        if execution_times:
            execution_times.sort(key=lambda x: x[1])
            metrics["fastest_linter"] = execution_times[0][0]
            metrics["slowest_linter"] = execution_times[-1][0]
            metrics["average_execution_time"] = sum(t[1] for t in execution_times) / len(execution_times)
        
        total_time = sum(t[1] for t in execution_times)
        if total_time > 0:
            metrics["issues_per_second"] = total_issues / total_time
        
        return metrics
    
    def _generate_linter_report(self, linter_results: Dict[str, Dict[str, Any]], directory_path: Path) -> Optional[Path]:
        """Generate a comprehensive linter report."""
        try:
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "directory": str(directory_path),
                "configuration": self.configuration,
                "linter_results": linter_results,
                "summary": self._generate_linter_summary(linter_results),
                "performance_metrics": self._calculate_performance_metrics(linter_results)
            }
            
            report_path = directory_path / "linter_report.json"
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=2)
            
            return report_path
        except Exception:
            return None
    
    def pre_execute(self, context) -> None:
        """Called before plugin execution."""
        print(f"Production Linter Runner: Processing directory {context.project_root}")
        available_linters = self._get_available_linters()
        print(f"Available linters: {available_linters}")
        print(f"Configuration: parallel={self.configuration.get('parallel_execution', True)}, "
              f"max_workers={self.configuration.get('max_workers', 4)}")
    
    def post_execute(self, context, result) -> None:
        """Called after plugin execution."""
        if result.success:
            print(f"Production Linter Runner: Found {result.issues_found} issues in {result.files_processed} files")
            
            # Print linter breakdown
            linter_results = result.get("linter_results", {})
            for linter_name, linter_result in linter_results.items():
                if linter_result.get("success", False):
                    issues = linter_result.get("issues_found", 0)
                    exec_time = linter_result.get("execution_time", 0.0)
                    print(f"  {linter_name}: {issues} issues in {exec_time:.2f}s")
        else:
            print(f"Production Linter Runner: Failed to process directory")
            if result.get("errors"):
                for error in result["errors"][:3]:  # Show first 3 errors
                    print(f"Error: {error}")
        
        if result.get("warnings"):
            for warning in result["warnings"][:3]:  # Show first 3 warnings
                print(f"Warning: {warning}")