from typing import Set, List, Dict, Any, Optional
from typing import List, Dict, Any, Optional
"""
FawltyDeps Analyzer Plugin

Plugin for analyzing Python dependencies using FawltyDeps.
FawltyDeps identifies undeclared and unused third-party dependencies.
"""

import json
import subprocess
import time

from .base_plugin import BasePlugin, PluginContext, PluginResult, PluginMetadata, PluginCategory, PluginPriority
import logging


class FawltyDepsAnalyzer(BasePlugin):
    """
    Plugin for analyzing dependencies using FawltyDeps.
    
    FawltyDeps identifies:
    - Undeclared dependencies (imports not in dependency files)
    - Unused dependencies (declared but not imported)
    """
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="fawltydeps_analyzer",
            version="1.0.0",
            description="Analyze Python dependencies using FawltyDeps",
            author="Code Quality Pipeline",
            category=PluginCategory.ANALYSIS,
            priority=PluginPriority.HIGH,
            dependencies=["fawltydeps"],
            tags={"dependency", "import", "analysis", "fawltydeps"},
            required_packages=["fawltydeps"],
            configuration_schema={
                "output_format": {
                    "type": "string",
                    "default": "json",
                    "enum": ["json", "human", "human_detailed"]
                },
                "ignore_unused": {
                    "type": "array",
                    "default": [],
                    "items": {"type": "string"}
                },
                "ignore_undeclared": {
                    "type": "array", 
                    "default": [],
                    "items": {"type": "string"}
                },
                "deps_files": {
                    "type": "array",
                    "default": ["pyproject.toml", "requirements.txt", "setup.py"],
                    "items": {"type": "string"}
                },
                "code_dirs": {
                    "type": "array",
                    "default": ["src", "."],
                    "items": {"type": "string"}
                }
            }
        )
    
    def is_available(self) -> bool:
        """Check if FawltyDeps is available."""
        try:
            result = subprocess.run(
                ["fawltydeps", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def execute(self, context: PluginContext) -> PluginResult:
        """Execute FawltyDeps analysis."""
        result = PluginResult(
            plugin_name=self.metadata.name,
            success=False,
            execution_time=0.0
        )
        
        start_time = time.time()
        
        try:
            # Prepare FawltyDeps command
            cmd = self._build_command(context)
            
            # Run FawltyDeps
            process_result = subprocess.run(
                cmd,
                cwd=context.project_root,
                capture_output=True,
                text=True,
                timeout=context.timeout
            )
            
            # Parse results
            analysis_results = self._parse_output(process_result, context)
            
            # Update result
            result.success = True
            result.execution_time = time.time() - start_time
            result.files_processed = len(context.target_files)
            result.issues_found = (
                len(analysis_results.get("undeclared_deps", [])) +
                len(analysis_results.get("unused_deps", []))
            )
            result.output_data = analysis_results
            
            # Add metrics
            result.add_metric("undeclared_count", len(analysis_results.get("undeclared_deps", [])))
            result.add_metric("unused_count", len(analysis_results.get("unused_deps", [])))
            result.add_metric("total_deps_analyzed", len(analysis_results.get("all_deps", [])))
            
            # Add warnings for issues found
            if analysis_results.get("undeclared_deps"):
                result.add_warning(f"Found {len(analysis_results['undeclared_deps'])} undeclared dependencies")
            if analysis_results.get("unused_deps"):
                result.add_warning(f"Found {len(analysis_results['unused_deps'])} unused dependencies")
                
        except subprocess.TimeoutExpired:
            result.add_error(f"FawltyDeps execution timed out after {context.timeout} seconds")
        except subprocess.CalledProcessError as e:
            result.add_error(f"FawltyDeps execution failed: {e.stderr}")
        except Exception as e:
            result.add_error(f"Unexpected error during FawltyDeps execution: {str(e)}")
        finally:
            result.execution_time = time.time() - start_time
        
        return result
    
    def _build_command(self, context: PluginContext) -> List[str]:
        """Build FawltyDeps command with configuration."""
        cmd = ["fawltydeps"]
        
        # Set output format
        output_format = self.configuration.get("output_format", "json")
        if output_format == "json":
            cmd.extend(["--output-format", "json"])
        elif output_format == "human_detailed":
            cmd.extend(["--output-format", "human_detailed"])
        
        # Add dependency files
        deps_files = self.configuration.get("deps_files", ["pyproject.toml", "requirements.txt"])
        for deps_file in deps_files:
            deps_path = context.project_root / deps_file
            if deps_path.exists():
                cmd.extend(["--deps", str(deps_path)])
        
        # Add code directories
        code_dirs = self.configuration.get("code_dirs", ["src", "."])
        for code_dir in code_dirs:
            code_path = context.project_root / code_dir
            if code_path.exists():
                cmd.extend(["--code", str(code_path)])
        
        # Add ignore patterns
        ignore_unused = self.configuration.get("ignore_unused", [])
        for dep in ignore_unused:
            cmd.extend(["--ignore-unused", dep])
        
        ignore_undeclared = self.configuration.get("ignore_undeclared", [])
        for dep in ignore_undeclared:
            cmd.extend(["--ignore-undeclared", dep])
        
        return cmd
    
    def _parse_output(self, process_result: subprocess.CompletedProcess, context: PluginContext) -> Dict[str, Any]:
        """Parse FawltyDeps output."""
        results = {
            "undeclared_deps": [],
            "unused_deps": [],
            "all_deps": [],
            "raw_output": process_result.stdout,
            "raw_errors": process_result.stderr,
            "return_code": process_result.returncode
        }
        
        if process_result.returncode != 0:
            return results
        
        # Try to parse JSON output
        if self.configuration.get("output_format", "json") == "json":
            try:
                json_data = json.loads(process_result.stdout)
                results.update(json_data)
            except json.JSONDecodeError:
                # Fall back to text parsing
                results.update(self._parse_text_output(process_result.stdout))
        else:
            # Parse text output
            results.update(self._parse_text_output(process_result.stdout))
        
        return results
    
    def _parse_text_output(self, output: str) -> Dict[str, Any]:
        """Parse text output from FawltyDeps."""
        undeclared = []
        unused = []
        
        lines = output.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if "undeclared dependencies" in line.lower():
                current_section = "undeclared"
            elif "unused dependencies" in line.lower():
                current_section = "unused"
            elif line.startswith("- ") and current_section:
                dep_name = line[2:].strip()
                if current_section == "undeclared":
                    undeclared.append(dep_name)
                elif current_section == "unused":
                    unused.append(dep_name)
        
        return {
            "undeclared_deps": undeclared,
            "unused_deps": unused
        }
    
    def get_supported_file_types(self) -> Set[str]:
        """Get supported file types."""
        return {'.py', '.pyi'}
    
    def validate_dependencies(self) -> List[str]:
        """Validate plugin dependencies."""
        missing = super().validate_dependencies()
        
        # Check if fawltydeps command is available
        try:
            subprocess.run(
                ["fawltydeps", "--version"],
                capture_output=True,
                timeout=5
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            missing.append("fawltydeps")
        
        return missing