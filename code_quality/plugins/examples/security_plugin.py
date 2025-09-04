"""
Security Scanner Plugin Example

Demonstrates how to create a plugin for security scanning functionality.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Set, List
from code_quality.plugins.base_plugin import DirectoryProcessorPlugin, PluginMetadata, PluginCategory, PluginPriority


class SecurityScannerPlugin(DirectoryProcessorPlugin):
    """
    Plugin for running security scanners on Python code.
    """
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="security_scanner",
            version="1.0.0",
            description="Runs security scanners on Python code",
            author="Code Quality Team",
            category=PluginCategory.SECURITY,
            priority=PluginPriority.HIGH,
            dependencies=[],
            tags={"security", "scanning", "python", "vulnerabilities"},
            required_packages=[],
            optional_packages=["bandit", "safety"],
            configuration_schema={
                "scanners": {"type": "list", "default": ["bandit"]},
                "severity_level": {"type": "string", "default": "medium"},
                "confidence_level": {"type": "string", "default": "medium"},
                "exclude_patterns": {"type": "list", "default": ["test_*.py", "*_test.py"]}
            }
        )
    
    def is_available(self) -> bool:
        """Check if plugin is available."""
        # Check if at least one security scanner is available
        available_scanners = []
        for scanner in self.configuration.get("scanners", ["bandit"]):
            try:
                subprocess.run([scanner, "--version"], 
                             capture_output=True, check=True, timeout=5)
                available_scanners.append(scanner)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        return len(available_scanners) > 0
    
    def get_supported_file_types(self) -> Set[str]:
        """Get supported file types."""
        return {'.py', '.pyi'}
    
    def process_directory(self, directory_path: Path, context) -> Dict[str, Any]:
        """
        Process a directory with security scanners.
        
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
            # Get available scanners
            available_scanners = self._get_available_scanners()
            if not available_scanners:
                result["success"] = False
                result["errors"].append("No security scanners available")
                return result
            
            # Find Python files
            python_files = self._find_python_files(directory_path)
            result["files_processed"] = len(python_files)
            
            if not python_files:
                return result
            
            # Run security scanners
            scanner_results = {}
            total_issues = 0
            
            for scanner in available_scanners:
                try:
                    scanner_result = self._run_scanner(scanner, python_files, context)
                    scanner_results[scanner] = scanner_result
                    total_issues += scanner_result.get("issues_found", 0)
                except Exception as e:
                    result["warnings"].append(f"Failed to run {scanner}: {e}")
            
            result["issues_found"] = total_issues
            result["output_data"] = {
                "scanner_results": scanner_results,
                "available_scanners": available_scanners,
                "security_summary": self._generate_security_summary(scanner_results)
            }
            
            # Note: Security scanners typically don't fix issues, they just report them
            result["issues_fixed"] = 0
        
        except Exception as e:
            result["success"] = False
            result["errors"].append(str(e))
        
        return result
    
    def _get_available_scanners(self) -> List[str]:
        """Get list of available security scanners."""
        available = []
        configured_scanners = self.configuration.get("scanners", ["bandit"])
        
        for scanner in configured_scanners:
            try:
                subprocess.run([scanner, "--version"], 
                             capture_output=True, check=True, timeout=5)
                available.append(scanner)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        return available
    
    def _find_python_files(self, directory_path: Path) -> List[Path]:
        """Find Python files in directory."""
        python_files = []
        exclude_patterns = self.configuration.get("exclude_patterns", ["test_*.py", "*_test.py"])
        
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
    
    def _run_scanner(self, scanner: str, files: List[Path], context) -> Dict[str, Any]:
        """
        Run a specific security scanner on files.
        
        Args:
            scanner: Name of the scanner to run
            files: List of files to scan
            context: Plugin execution context
            
        Returns:
            Dict[str, Any]: Scanner result
        """
        result = {
            "scanner": scanner,
            "issues_found": 0,
            "output": "",
            "errors": []
        }
        
        try:
            # Build command based on scanner
            if scanner == "bandit":
                cmd = self._build_bandit_command(files)
            elif scanner == "safety":
                cmd = self._build_safety_command()
            else:
                result["errors"].append(f"Unknown scanner: {scanner}")
                return result
            
            # Run the scanner
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
                result["issues_found"] = self._count_security_issues(result["output"], scanner)
            else:
                result["output"] = f"Dry run: Would execute {' '.join(cmd)}"
                result["issues_found"] = 0
        
        except subprocess.TimeoutExpired:
            result["errors"].append(f"Scanner {scanner} timed out")
        except Exception as e:
            result["errors"].append(f"Error running {scanner}: {e}")
        
        return result
    
    def _build_bandit_command(self, files: List[Path]) -> List[str]:
        """Build bandit command."""
        cmd = ["bandit"]
        
        # Add configuration options
        severity_level = self.configuration.get("severity_level", "medium")
        confidence_level = self.configuration.get("confidence_level", "medium")
        
        cmd.extend(["-ll"])  # Low severity, low confidence by default
        
        if severity_level == "high":
            cmd.extend(["-lll"])  # High severity
        elif severity_level == "medium":
            cmd.extend(["-ll"])   # Medium severity
        
        # Add format option for better parsing
        cmd.extend(["-f", "json"])
        
        # Add files
        cmd.extend([str(f) for f in files])
        
        return cmd
    
    def _build_safety_command(self) -> List[str]:
        """Build safety command."""
        cmd = ["safety", "check"]
        
        # Add JSON output for better parsing
        cmd.extend(["--json"])
        
        return cmd
    
    def _count_security_issues(self, output: str, scanner: str) -> int:
        """Count security issues in scanner output."""
        if not output:
            return 0
        
        if scanner == "bandit":
            # Try to parse JSON output
            try:
                import json
                data = json.loads(output)
                if isinstance(data, dict) and "results" in data:
                    return len(data["results"])
            except (json.JSONDecodeError, KeyError):
                pass
            
            # Fallback: count lines with issue indicators
            lines = output.strip().split('\n')
            issue_count = 0
            for line in lines:
                if any(indicator in line.lower() for indicator in ["high", "medium", "low", "issue", "vulnerability"]):
                    issue_count += 1
            return issue_count
        
        elif scanner == "safety":
            # Try to parse JSON output
            try:
                import json
                data = json.loads(output)
                if isinstance(data, list):
                    return len(data)
            except (json.JSONDecodeError, KeyError):
                pass
            
            # Fallback: count vulnerability lines
            lines = output.strip().split('\n')
            issue_count = 0
            for line in lines:
                if "vulnerability" in line.lower() or "CVE" in line:
                    issue_count += 1
            return issue_count
        
        return 0
    
    def _generate_security_summary(self, scanner_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a security summary from scanner results."""
        summary = {
            "total_issues": 0,
            "high_severity": 0,
            "medium_severity": 0,
            "low_severity": 0,
            "scanners_used": list(scanner_results.keys()),
            "recommendations": []
        }
        
        for scanner, result in scanner_results.items():
            issues = result.get("issues_found", 0)
            summary["total_issues"] += issues
            
            # Basic severity classification (could be enhanced with actual parsing)
            if issues > 0:
                if scanner == "bandit":
                    summary["medium_severity"] += issues
                elif scanner == "safety":
                    summary["high_severity"] += issues
        
        # Generate recommendations
        if summary["high_severity"] > 0:
            summary["recommendations"].append("Address high-severity security issues immediately")
        if summary["medium_severity"] > 0:
            summary["recommendations"].append("Review and fix medium-severity security issues")
        if summary["total_issues"] == 0:
            summary["recommendations"].append("No security issues found - good job!")
        
        return summary
    
    def pre_execute(self, context) -> None:
        """Called before plugin execution."""
        print(f"Security Scanner Plugin: Scanning directory {context.project_root}")
    
    def post_execute(self, context, result) -> None:
        """Called after plugin execution."""
        if result.success:
            print(f"Security Scanner Plugin: Found {result.issues_found} security issues in {result.files_processed} files")
        else:
            print(f"Security Scanner Plugin: Failed to scan directory")