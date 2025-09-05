from typing import Set, List, Dict, Any, Optional
from typing import List, Dict, Any, Optional
"""
Creosote Analyzer Plugin

Plugin for analyzing unused dependencies using Creosote.
Creosote identifies unused dependencies in Python projects.
"""

import json
import subprocess
import time

from .base_plugin import BasePlugin, PluginContext, PluginResult, PluginMetadata, PluginCategory, PluginPriority
import logging


class CreosoteAnalyzer(BasePlugin):
    """
    Plugin for analyzing unused dependencies using Creosote.
    
    Creosote identifies:
    - Unused dependencies (declared but not imported)
    - Dependencies that can be safely removed
    """
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="creosote_analyzer",
            version="1.0.0",
            description="Analyze unused dependencies using Creosote",
            author="Code Quality Pipeline",
            category=PluginCategory.ANALYSIS,
            priority=PluginPriority.HIGH,
            dependencies=["creosote"],
            tags={"dependency", "unused", "analysis", "creosote"},
            required_packages=["creosote"],
            configuration_schema={
                "venv_path": {
                    "type": "string",
                    "default": ".venv",
                    "description": "Path to virtual environment"
                },
                "project_path": {
                    "type": "string",
                    "default": "src",
                    "description": "Path to project source code"
                },
                "deps_file": {
                    "type": "string",
                    "default": "pyproject.toml",
                    "description": "Dependency file to analyze"
                },
                "section": {
                    "type": "string",
                    "default": "project.dependencies",
                    "description": "Section in dependency file"
                },
                "exclude": {
                    "type": "array",
                    "default": [],
                    "items": {"type": "string"},
                    "description": "Dependencies to exclude from analysis"
                },
                "output_format": {
                    "type": "string",
                    "default": "json",
                    "enum": ["json", "text"]
                }
            }
        )
    
    def is_available(self) -> bool:
        """Check if Creosote is available."""
        try:
            result = subprocess.run(
                ["creosote", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def execute(self, context: PluginContext) -> PluginResult:
        """Execute Creosote analysis."""
        result = PluginResult(
            plugin_name=self.metadata.name,
            success=False,
            execution_time=0.0
        )
        
        start_time = time.time()
        
        try:
            # Prepare Creosote command
            cmd = self._build_command(context)
            
            # Run Creosote
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
            result.issues_found = len(analysis_results.get("unused_deps", []))
            result.output_data = analysis_results
            
            # Add metrics
            result.add_metric("unused_count", len(analysis_results.get("unused_deps", [])))
            result.add_metric("total_deps_checked", len(analysis_results.get("all_deps", [])))
            result.add_metric("excluded_count", len(analysis_results.get("excluded_deps", [])))
            
            # Add warnings for unused dependencies found
            if analysis_results.get("unused_deps"):
                result.add_warning(f"Found {len(analysis_results['unused_deps'])} unused dependencies")
                
        except subprocess.TimeoutExpired:
            result.add_error(f"Creosote execution timed out after {context.timeout} seconds")
        except subprocess.CalledProcessError as e:
            result.add_error(f"Creosote execution failed: {e.stderr}")
        except Exception as e:
            result.add_error(f"Unexpected error during Creosote execution: {str(e)}")
        finally:
            result.execution_time = time.time() - start_time
        
        return result
    
    def _build_command(self, context: PluginContext) -> List[str]:
        """Build Creosote command with configuration."""
        cmd = ["creosote"]
        
        # Set virtual environment path
        venv_path = self.configuration.get("venv_path", ".venv")
        venv_full_path = context.project_root / venv_path
        if venv_full_path.exists():
            cmd.extend(["--venv", str(venv_full_path)])
        
        # Set project path
        project_path = self.configuration.get("project_path", "src")
        project_full_path = context.project_root / project_path
        if project_full_path.exists():
            cmd.extend(["--path", str(project_full_path)])
        
        # Set dependency file
        deps_file = self.configuration.get("deps_file", "pyproject.toml")
        deps_full_path = context.project_root / deps_file
        if deps_full_path.exists():
            cmd.extend(["--deps-file", str(deps_full_path)])
        
        # Set section
        section = self.configuration.get("section", "project.dependencies")
        cmd.extend(["--section", section])
        
        # Add exclusions
        exclude_deps = self.configuration.get("exclude", [])
        for dep in exclude_deps:
            cmd.extend(["--exclude", dep])
        
        # Set output format
        output_format = self.configuration.get("output_format", "json")
        if output_format == "json":
            cmd.append("--json")
        
        return cmd
    
    def _parse_output(self, process_result: subprocess.CompletedProcess, context: PluginContext) -> Dict[str, Any]:
        """Parse Creosote output."""
        results = {
            "unused_deps": [],
            "all_deps": [],
            "excluded_deps": [],
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
        """Parse text output from Creosote."""
        unused = []
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Creosote typically outputs one unused dependency per line
            if line and not line.startswith('Found') and not line.startswith('Checking'):
                # Clean up the line (remove any extra formatting)
                dep_name = line.split()[0] if line.split() else line
                if dep_name:
                    unused.append(dep_name)
        
        return {
            "unused_deps": unused
        }
    
    def get_supported_file_types(self) -> Set[str]:
        """Get supported file types."""
        return {'.py', '.pyi'}
    
    def validate_dependencies(self) -> List[str]:
        """Validate plugin dependencies."""
        missing = super().validate_dependencies()
        
        # Check if creosote command is available
        try:
            subprocess.run(
                ["creosote", "--version"],
                capture_output=True,
                timeout=5
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            missing.append("creosote")
        
        return missing