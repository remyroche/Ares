"""
Linter Plugin Example

Demonstrates how to create a plugin for linting functionality.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Set, List
from code_quality.plugins.base_plugin import DirectoryProcessorPlugin, PluginMetadata, PluginCategory, PluginPriority


class LinterPlugin(DirectoryProcessorPlugin):
    """
    Plugin for running linters on Python code.
    """
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="linter",
            version="1.0.0",
            description="Runs linters on Python code",
            author="Code Quality Team",
            category=PluginCategory.LINTING,
            priority=PluginPriority.MEDIUM,
            dependencies=[],
            tags={"linting", "analysis", "python"},
            required_packages=[],
            optional_packages=["flake8", "pylint", "mypy"],
            configuration_schema={
                "linters": {"type": "list", "default": ["flake8"]},
                "max_line_length": {"type": "integer", "default": 120},
                "ignore_errors": {"type": "list", "default": []},
                "exclude_patterns": {"type": "list", "default": ["__pycache__", "*.pyc"]}
            }
        )
    
    def is_available(self) -> bool:
        """Check if plugin is available."""
        # Check if at least one linter is available
        available_linters = []
        for linter in self.configuration.get("linters", ["flake8"]):
            try:
                subprocess.run([linter, "--version"], 
                             capture_output=True, check=True, timeout=5)
                available_linters.append(linter)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        return len(available_linters) > 0
    
    def get_supported_file_types(self) -> Set[str]:
        """Get supported file types."""
        return {'.py', '.pyi'}
    
    def process_directory(self, directory_path: Path, context) -> Dict[str, Any]:
        """
        Process a directory with linters.
        
        Args:
            directory_path: Path to the directory to process
            context: Plugin execution context
            
        Returns:
            Dict[str, Any]: Processing result
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
            "output_data": {}
        }
        
        try:
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
                return result
            
            # Run linters
            linter_results = {}
            total_issues = 0
            
            for linter in available_linters:
                try:
                    linter_result = self._run_linter(linter, python_files, context)
                    linter_results[linter] = linter_result
                    total_issues += linter_result.get("issues_found", 0)
                except Exception as e:
                    result["warnings"].append(f"Failed to run {linter}: {e}")
            
            result["issues_found"] = total_issues
            result["output_data"] = {
                "linter_results": linter_results,
                "available_linters": available_linters
            }
            
            # Note: Linters typically don't fix issues, they just report them
            result["issues_fixed"] = 0
        
        except Exception as e:
            result["success"] = False
            result["errors"].append(str(e))
        
        return result
    
    def _get_available_linters(self) -> List[str]:
        """Get list of available linters."""
        available = []
        configured_linters = self.configuration.get("linters", ["flake8"])
        
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
        exclude_patterns = self.configuration.get("exclude_patterns", ["__pycache__", "*.pyc"])
        
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
    
    def _run_linter(self, linter: str, files: List[Path], context) -> Dict[str, Any]:
        """
        Run a specific linter on files.
        
        Args:
            linter: Name of the linter to run
            files: List of files to lint
            context: Plugin execution context
            
        Returns:
            Dict[str, Any]: Linter result
        """
        result = {
            "linter": linter,
            "issues_found": 0,
            "output": "",
            "errors": []
        }
        
        try:
            # Build command based on linter
            if linter == "flake8":
                cmd = self._build_flake8_command(files)
            elif linter == "pylint":
                cmd = self._build_pylint_command(files)
            elif linter == "mypy":
                cmd = self._build_mypy_command(files)
            else:
                result["errors"].append(f"Unknown linter: {linter}")
                return result
            
            # Run the linter
            if not context.dry_run:
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=context.timeout,
                    cwd=str(context.project_root)
                )
                
                result["output"] = process.stdout + process.stderr
                result["return_code"] = process.returncode
                
                # Count issues (basic parsing)
                result["issues_found"] = self._count_issues(result["output"], linter)
            else:
                result["output"] = f"Dry run: Would execute {' '.join(cmd)}"
                result["issues_found"] = 0
        
        except subprocess.TimeoutExpired:
            result["errors"].append(f"Linter {linter} timed out")
        except Exception as e:
            result["errors"].append(f"Error running {linter}: {e}")
        
        return result
    
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
        
        # Add files
        cmd.extend([str(f) for f in files])
        
        return cmd
    
    def _build_pylint_command(self, files: List[Path]) -> List[str]:
        """Build pylint command."""
        cmd = ["pylint"]
        
        # Add configuration options
        max_line_length = self.configuration.get("max_line_length", 120)
        cmd.extend(["--max-line-length", str(max_line_length)])
        
        # Add files
        cmd.extend([str(f) for f in files])
        
        return cmd
    
    def _build_mypy_command(self, files: List[Path]) -> List[str]:
        """Build mypy command."""
        cmd = ["mypy"]
        
        # Add files
        cmd.extend([str(f) for f in files])
        
        return cmd
    
    def _count_issues(self, output: str, linter: str) -> int:
        """Count issues in linter output."""
        if not output:
            return 0
        
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
        
        return issue_count
    
    def pre_execute(self, context) -> None:
        """Called before plugin execution."""
        print(f"Linter Plugin: Processing directory {context.project_root}")
    
    def post_execute(self, context, result) -> None:
        """Called after plugin execution."""
        if result.success:
            print(f"Linter Plugin: Found {result.issues_found} issues in {result.files_processed} files")
        else:
            print(f"Linter Plugin: Failed to process directory")