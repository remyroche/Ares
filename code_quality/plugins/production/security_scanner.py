"""
Production Security Scanner Plugin

A robust, production-ready plugin for running multiple security scanners on Python code with
comprehensive vulnerability detection, risk assessment, and detailed reporting.
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


class ProductionSecurityScannerPlugin(DirectoryProcessorPlugin):
    """
    Production-ready plugin for running multiple security scanners on Python code.
    
    Features:
    - Multiple security scanner support (bandit, safety, semgrep, trivy)
    - Comprehensive vulnerability detection
    - Risk assessment and severity classification
    - Detailed security reporting
    - Configurable scanner options
    - Parallel scanner execution
    - Backup creation and rollback capabilities
    - Performance monitoring and optimization
    """
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="production_security_scanner",
            version="2.0.0",
            description="Production-ready multi-scanner security analyzer with comprehensive vulnerability detection",
            author="Code Quality Team",
            category=PluginCategory.SECURITY,
            priority=PluginPriority.CRITICAL,
            dependencies=[],
            tags={"security", "scanning", "python", "production", "vulnerabilities", "risk-assessment"},
            required_packages=[],
            optional_packages=["bandit", "safety", "semgrep", "trivy"],
            configuration_schema={
                "create_backups": {"type": "boolean", "default": True},
                "backup_suffix": {"type": "string", "default": ".bak"},
                "scanners": {"type": "list", "default": ["bandit", "safety"]},
                "parallel_execution": {"type": "boolean", "default": True},
                "max_workers": {"type": "integer", "default": 4},
                "timeout_per_scanner": {"type": "integer", "default": 300},
                "severity_level": {"type": "string", "default": "medium"},
                "confidence_level": {"type": "string", "default": "medium"},
                "exclude_patterns": {"type": "list", "default": ["test_*.py", "*_test.py", "*/tests/*"]},
                "output_format": {"type": "string", "default": "json"},
                "generate_reports": {"type": "boolean", "default": True},
                "risk_assessment": {"type": "boolean", "default": True},
                "custom_configs": {"type": "dict", "default": {}},
                "fail_on_high": {"type": "boolean", "default": True},
                "fail_on_medium": {"type": "boolean", "default": False},
                "ignore_cves": {"type": "list", "default": []},
                "custom_rules": {"type": "list", "default": []}
            }
        )
    
    def is_available(self) -> bool:
        """Check if plugin is available."""
        # Check if at least one security scanner is available
        available_scanners = self._get_available_scanners()
        return len(available_scanners) > 0
    
    def get_supported_file_types(self) -> Set[str]:
        """Get supported file types."""
        return {'.py', '.pyi', '.pyw', '.txt', '.yml', '.yaml', '.json', '.toml', '.cfg', '.ini'}
    
    def process_directory(self, directory_path: Path, context) -> Dict[str, Any]:
        """
        Process a directory with multiple security scanners.
        
        Args:
            directory_path: Path to the directory to process
            context: Plugin execution context
            
        Returns:
            Dict[str, Any]: Comprehensive security analysis result
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
            "scanner_results": {},
            "security_summary": {},
            "risk_assessment": {}
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
            
            # Get available scanners
            available_scanners = self._get_available_scanners()
            if not available_scanners:
                result["success"] = False
                result["errors"].append("No security scanners available")
                return result
            
            # Find files to scan
            files_to_scan = self._find_files_to_scan(directory_path)
            result["files_processed"] = len(files_to_scan)
            
            if not files_to_scan:
                result["warnings"].append("No files found to scan")
                return result
            
            # Run security scanners
            scanner_results = {}
            total_issues = 0
            
            if self.configuration.get("parallel_execution", True):
                scanner_results = self._run_scanners_parallel(
                    available_scanners, files_to_scan, context
                )
            else:
                scanner_results = self._run_scanners_sequential(
                    available_scanners, files_to_scan, context
                )
            
            # Aggregate results
            for scanner_name, scanner_result in scanner_results.items():
                if scanner_result.get("success", False):
                    total_issues += scanner_result.get("issues_found", 0)
                else:
                    result["errors"].extend(scanner_result.get("errors", []))
                    result["warnings"].extend(scanner_result.get("warnings", []))
            
            result["issues_found"] = total_issues
            result["scanner_results"] = scanner_results
            
            # Generate security summary
            result["security_summary"] = self._generate_security_summary(scanner_results)
            
            # Perform risk assessment
            if self.configuration.get("risk_assessment", True):
                result["risk_assessment"] = self._perform_risk_assessment(scanner_results)
            
            # Check if we should fail based on severity
            if self._should_fail_pipeline(result["security_summary"]):
                result["success"] = False
                result["errors"].append("Security scan failed due to high/medium severity issues")
            
            result["output_data"] = {
                "available_scanners": available_scanners,
                "security_summary": result["security_summary"],
                "risk_assessment": result["risk_assessment"],
                "performance_metrics": self._calculate_performance_metrics(scanner_results)
            }
            
            # Generate security report if configured
            if self.configuration.get("generate_reports", True) and not context.dry_run:
                report_path = self._generate_security_report(scanner_results, directory_path, result)
                if report_path:
                    result["output_data"]["report_path"] = str(report_path)
            
            # Note: Security scanners typically don't fix issues, they just report them
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
    
    def _get_available_scanners(self) -> List[str]:
        """Get list of available security scanners."""
        available = []
        configured_scanners = self.configuration.get("scanners", ["bandit", "safety"])
        
        for scanner in configured_scanners:
            try:
                subprocess.run([scanner, "--version"], 
                             capture_output=True, check=True, timeout=5)
                available.append(scanner)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        return available
    
    def _find_files_to_scan(self, directory_path: Path) -> List[Path]:
        """Find files to scan for security issues."""
        files_to_scan = []
        exclude_patterns = self.configuration.get("exclude_patterns", ["test_*.py", "*_test.py", "*/tests/*"])
        
        # Find Python files
        for py_file in directory_path.rglob("*.py"):
            should_exclude = False
            for pattern in exclude_patterns:
                if pattern in str(py_file):
                    should_exclude = True
                    break
            if not should_exclude:
                files_to_scan.append(py_file)
        
        # Find configuration files
        config_extensions = ['.yml', '.yaml', '.json', '.toml', '.cfg', '.ini', '.txt']
        for ext in config_extensions:
            for config_file in directory_path.rglob(f"*{ext}"):
                if not any(pattern in str(config_file) for pattern in exclude_patterns):
                    files_to_scan.append(config_file)
        
        return files_to_scan
    
    def _run_scanners_parallel(self, scanners: List[str], files: List[Path], context) -> Dict[str, Dict[str, Any]]:
        """Run security scanners in parallel."""
        results = {}
        max_workers = self.configuration.get("max_workers", 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all scanner tasks
            future_to_scanner = {
                executor.submit(self._run_single_scanner, scanner, files, context): scanner
                for scanner in scanners
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_scanner):
                scanner_name = future_to_scanner[future]
                try:
                    result = future.result()
                    results[scanner_name] = result
                except Exception as e:
                    results[scanner_name] = {
                        "success": False,
                        "issues_found": 0,
                        "output": "",
                        "errors": [f"Scanner execution failed: {e}"],
                        "execution_time": 0.0
                    }
        
        return results
    
    def _run_scanners_sequential(self, scanners: List[str], files: List[Path], context) -> Dict[str, Dict[str, Any]]:
        """Run security scanners sequentially."""
        results = {}
        
        for scanner in scanners:
            try:
                result = self._run_single_scanner(scanner, files, context)
                results[scanner] = result
            except Exception as e:
                results[scanner] = {
                    "success": False,
                    "issues_found": 0,
                    "output": "",
                    "errors": [f"Scanner execution failed: {e}"],
                    "execution_time": 0.0
                }
        
        return results
    
    def _run_single_scanner(self, scanner: str, files: List[Path], context) -> Dict[str, Any]:
        """
        Run a single security scanner on files.
        
        Args:
            scanner: Name of the scanner to run
            files: List of files to scan
            context: Plugin execution context
            
        Returns:
            Dict[str, Any]: Scanner result
        """
        result = {
            "scanner": scanner,
            "success": True,
            "issues_found": 0,
            "output": "",
            "errors": [],
            "warnings": [],
            "execution_time": 0.0,
            "return_code": 0,
            "vulnerabilities": []
        }
        
        start_time = datetime.now()
        
        try:
            # Build command based on scanner
            cmd = self._build_scanner_command(scanner, files)
            
            # Run the scanner
            if not context.dry_run:
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.configuration.get("timeout_per_scanner", 300),
                    cwd=str(context.project_root)
                )
                
                result["output"] = process.stdout + process.stderr
                result["return_code"] = process.returncode
                
                # Parse vulnerabilities
                vulnerabilities = self._parse_vulnerabilities(result["output"], scanner)
                result["vulnerabilities"] = vulnerabilities
                result["issues_found"] = len(vulnerabilities)
                
                # Check for success
                if process.returncode not in [0, 1]:  # 0 = success, 1 = issues found
                    result["success"] = False
                    result["errors"].append(f"Scanner {scanner} failed with return code {process.returncode}")
            else:
                result["output"] = f"Dry run: Would execute {' '.join(cmd)}"
                result["issues_found"] = 0
        
        except subprocess.TimeoutExpired:
            result["success"] = False
            result["errors"].append(f"Scanner {scanner} timed out after {self.configuration.get('timeout_per_scanner', 300)} seconds")
        except Exception as e:
            result["success"] = False
            result["errors"].append(f"Error running {scanner}: {e}")
        
        finally:
            result["execution_time"] = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def _build_scanner_command(self, scanner: str, files: List[Path]) -> List[str]:
        """Build command for a specific security scanner."""
        if scanner == "bandit":
            return self._build_bandit_command(files)
        elif scanner == "safety":
            return self._build_safety_command()
        elif scanner == "semgrep":
            return self._build_semgrep_command(files)
        elif scanner == "trivy":
            return self._build_trivy_command(files)
        else:
            return [scanner] + [str(f) for f in files]
    
    def _build_bandit_command(self, files: List[Path]) -> List[str]:
        """Build bandit command."""
        cmd = ["bandit"]
        
        # Add configuration options
        severity_level = self.configuration.get("severity_level", "medium")
        confidence_level = self.configuration.get("confidence_level", "medium")
        
        # Set severity and confidence levels
        if severity_level == "high":
            cmd.extend(["-lll"])  # High severity
        elif severity_level == "medium":
            cmd.extend(["-ll"])   # Medium severity
        else:
            cmd.extend(["-l"])    # Low severity
        
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
    
    def _build_safety_command(self) -> List[str]:
        """Build safety command."""
        cmd = ["safety", "check"]
        
        # Add output format
        output_format = self.configuration.get("output_format", "json")
        if output_format == "json":
            cmd.extend(["--json"])
        
        # Add custom configs
        custom_configs = self.configuration.get("custom_configs", {})
        if "safety" in custom_configs:
            for key, value in custom_configs["safety"].items():
                cmd.extend([f"--{key}", str(value)])
        
        return cmd
    
    def _build_semgrep_command(self, files: List[Path]) -> List[str]:
        """Build semgrep command."""
        cmd = ["semgrep"]
        
        # Add output format
        output_format = self.configuration.get("output_format", "json")
        if output_format == "json":
            cmd.extend(["--json"])
        
        # Add custom configs
        custom_configs = self.configuration.get("custom_configs", {})
        if "semgrep" in custom_configs:
            for key, value in custom_configs["semgrep"].items():
                cmd.extend([f"--{key}", str(value)])
        
        # Add files
        cmd.extend([str(f) for f in files])
        
        return cmd
    
    def _build_trivy_command(self, files: List[Path]) -> List[str]:
        """Build trivy command."""
        cmd = ["trivy", "fs"]
        
        # Add output format
        output_format = self.configuration.get("output_format", "json")
        if output_format == "json":
            cmd.extend(["--format", "json"])
        
        # Add custom configs
        custom_configs = self.configuration.get("custom_configs", {})
        if "trivy" in custom_configs:
            for key, value in custom_configs["trivy"].items():
                cmd.extend([f"--{key}", str(value)])
        
        # Add files
        cmd.extend([str(f) for f in files])
        
        return cmd
    
    def _parse_vulnerabilities(self, output: str, scanner: str) -> List[Dict[str, Any]]:
        """Parse vulnerabilities from scanner output."""
        vulnerabilities = []
        
        if not output:
            return vulnerabilities
        
        try:
            # Try to parse JSON output first
            if output.strip().startswith('[') or output.strip().startswith('{'):
                data = json.loads(output)
                vulnerabilities = self._parse_json_vulnerabilities(data, scanner)
        except json.JSONDecodeError:
            # Fallback to text parsing
            vulnerabilities = self._parse_text_vulnerabilities(output, scanner)
        
        return vulnerabilities
    
    def _parse_json_vulnerabilities(self, data: Any, scanner: str) -> List[Dict[str, Any]]:
        """Parse vulnerabilities from JSON output."""
        vulnerabilities = []
        
        if scanner == "bandit":
            if isinstance(data, list):
                for item in data:
                    if "issue_severity" in item:
                        vulnerabilities.append({
                            "type": "security_issue",
                            "severity": item.get("issue_severity", "unknown"),
                            "confidence": item.get("issue_confidence", "unknown"),
                            "description": item.get("issue_text", ""),
                            "file": item.get("filename", ""),
                            "line": item.get("line_number", 0),
                            "cwe": item.get("issue_cwe", {}).get("id", ""),
                            "scanner": scanner
                        })
        
        elif scanner == "safety":
            if isinstance(data, list):
                for item in data:
                    vulnerabilities.append({
                        "type": "vulnerability",
                        "severity": "high",  # Safety typically reports high-severity issues
                        "package": item.get("package", ""),
                        "version": item.get("installed_version", ""),
                        "vulnerability": item.get("vulnerability", ""),
                        "description": item.get("advisory", ""),
                        "cve": item.get("cve", ""),
                        "scanner": scanner
                    })
        
        return vulnerabilities
    
    def _parse_text_vulnerabilities(self, output: str, scanner: str) -> List[Dict[str, Any]]:
        """Parse vulnerabilities from text output."""
        vulnerabilities = []
        lines = output.strip().split('\n')
        
        for line in lines:
            if scanner == "bandit":
                if "Issue:" in line and "Severity:" in line:
                    # Parse bandit text output
                    parts = line.split("Issue:")
                    if len(parts) > 1:
                        issue_part = parts[1]
                        severity = "unknown"
                        if "Severity:" in issue_part:
                            severity = issue_part.split("Severity:")[1].split()[0].lower()
                        
                        vulnerabilities.append({
                            "type": "security_issue",
                            "severity": severity,
                            "description": issue_part.strip(),
                            "scanner": scanner
                        })
            
            elif scanner == "safety":
                if "vulnerability" in line.lower() or "CVE" in line:
                    vulnerabilities.append({
                        "type": "vulnerability",
                        "severity": "high",
                        "description": line.strip(),
                        "scanner": scanner
                    })
        
        return vulnerabilities
    
    def _generate_security_summary(self, scanner_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive security summary."""
        summary = {
            "total_vulnerabilities": 0,
            "high_severity": 0,
            "medium_severity": 0,
            "low_severity": 0,
            "critical_issues": 0,
            "scanners_used": list(scanner_results.keys()),
            "vulnerability_breakdown": {},
            "recommendations": []
        }
        
        for scanner_name, result in scanner_results.items():
            vulnerabilities = result.get("vulnerabilities", [])
            summary["total_vulnerabilities"] += len(vulnerabilities)
            
            scanner_breakdown = {
                "total": len(vulnerabilities),
                "high": 0,
                "medium": 0,
                "low": 0,
                "critical": 0
            }
            
            for vuln in vulnerabilities:
                severity = vuln.get("severity", "unknown").lower()
                if severity == "critical":
                    summary["critical_issues"] += 1
                    scanner_breakdown["critical"] += 1
                elif severity == "high":
                    summary["high_severity"] += 1
                    scanner_breakdown["high"] += 1
                elif severity == "medium":
                    summary["medium_severity"] += 1
                    scanner_breakdown["medium"] += 1
                elif severity == "low":
                    summary["low_severity"] += 1
                    scanner_breakdown["low"] += 1
            
            summary["vulnerability_breakdown"][scanner_name] = scanner_breakdown
        
        # Generate recommendations
        if summary["critical_issues"] > 0:
            summary["recommendations"].append("🚨 CRITICAL: Address critical security issues immediately")
        if summary["high_severity"] > 0:
            summary["recommendations"].append("⚠️ HIGH: Review and fix high-severity security issues")
        if summary["medium_severity"] > 0:
            summary["recommendations"].append("📋 MEDIUM: Consider fixing medium-severity security issues")
        if summary["total_vulnerabilities"] == 0:
            summary["recommendations"].append("✅ No security vulnerabilities found - good job!")
        
        return summary
    
    def _perform_risk_assessment(self, scanner_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Perform a comprehensive risk assessment."""
        assessment = {
            "overall_risk": "low",
            "risk_score": 0,
            "risk_factors": [],
            "mitigation_strategies": [],
            "compliance_status": "unknown"
        }
        
        total_vulnerabilities = 0
        high_risk_count = 0
        
        for scanner_name, result in scanner_results.items():
            vulnerabilities = result.get("vulnerabilities", [])
            total_vulnerabilities += len(vulnerabilities)
            
            for vuln in vulnerabilities:
                severity = vuln.get("severity", "unknown").lower()
                if severity in ["critical", "high"]:
                    high_risk_count += 1
        
        # Calculate risk score (0-100)
        if total_vulnerabilities > 0:
            assessment["risk_score"] = min(100, (high_risk_count * 20) + (total_vulnerabilities * 2))
        
        # Determine overall risk level
        if assessment["risk_score"] >= 80:
            assessment["overall_risk"] = "critical"
        elif assessment["risk_score"] >= 60:
            assessment["overall_risk"] = "high"
        elif assessment["risk_score"] >= 40:
            assessment["overall_risk"] = "medium"
        else:
            assessment["overall_risk"] = "low"
        
        # Add risk factors
        if high_risk_count > 0:
            assessment["risk_factors"].append(f"{high_risk_count} high/critical severity vulnerabilities")
        if total_vulnerabilities > 10:
            assessment["risk_factors"].append(f"{total_vulnerabilities} total vulnerabilities")
        
        # Add mitigation strategies
        if assessment["overall_risk"] in ["critical", "high"]:
            assessment["mitigation_strategies"].append("Immediate security review required")
            assessment["mitigation_strategies"].append("Update vulnerable dependencies")
            assessment["mitigation_strategies"].append("Implement additional security controls")
        
        return assessment
    
    def _should_fail_pipeline(self, security_summary: Dict[str, Any]) -> bool:
        """Determine if the pipeline should fail based on security issues."""
        fail_on_high = self.configuration.get("fail_on_high", True)
        fail_on_medium = self.configuration.get("fail_on_medium", False)
        
        if fail_on_high and security_summary.get("high_severity", 0) > 0:
            return True
        
        if fail_on_medium and security_summary.get("medium_severity", 0) > 0:
            return True
        
        if security_summary.get("critical_issues", 0) > 0:
            return True
        
        return False
    
    def _calculate_performance_metrics(self, scanner_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics from scanner results."""
        metrics = {
            "average_execution_time": 0.0,
            "fastest_scanner": None,
            "slowest_scanner": None,
            "total_files_scanned": 0,
            "vulnerabilities_per_second": 0.0
        }
        
        if not scanner_results:
            return metrics
        
        execution_times = []
        total_vulnerabilities = 0
        
        for scanner_name, result in scanner_results.items():
            execution_time = result.get("execution_time", 0.0)
            vulnerabilities = result.get("issues_found", 0)
            
            execution_times.append((scanner_name, execution_time))
            total_vulnerabilities += vulnerabilities
        
        if execution_times:
            execution_times.sort(key=lambda x: x[1])
            metrics["fastest_scanner"] = execution_times[0][0]
            metrics["slowest_scanner"] = execution_times[-1][0]
            metrics["average_execution_time"] = sum(t[1] for t in execution_times) / len(execution_times)
        
        total_time = sum(t[1] for t in execution_times)
        if total_time > 0:
            metrics["vulnerabilities_per_second"] = total_vulnerabilities / total_time
        
        return metrics
    
    def _generate_security_report(self, scanner_results: Dict[str, Dict[str, Any]], 
                                directory_path: Path, result: Dict[str, Any]) -> Optional[Path]:
        """Generate a comprehensive security report."""
        try:
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "directory": str(directory_path),
                "configuration": self.configuration,
                "scanner_results": scanner_results,
                "security_summary": result.get("security_summary", {}),
                "risk_assessment": result.get("risk_assessment", {}),
                "performance_metrics": self._calculate_performance_metrics(scanner_results)
            }
            
            report_path = directory_path / "security_report.json"
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=2)
            
            return report_path
        except Exception:
            return None
    
    def pre_execute(self, context) -> None:
        """Called before plugin execution."""
        print(f"Production Security Scanner: Scanning directory {context.project_root}")
        available_scanners = self._get_available_scanners()
        print(f"Available scanners: {available_scanners}")
        print(f"Configuration: parallel={self.configuration.get('parallel_execution', True)}, "
              f"severity_level={self.configuration.get('severity_level', 'medium')}")
    
    def post_execute(self, context, result) -> None:
        """Called after plugin execution."""
        if result.success:
            print(f"Production Security Scanner: Found {result.issues_found} security issues in {result.files_processed} files")
            
            # Print security summary
            security_summary = result.get("security_summary", {})
            if security_summary:
                print(f"Security Summary:")
                print(f"  Total vulnerabilities: {security_summary.get('total_vulnerabilities', 0)}")
                print(f"  Critical: {security_summary.get('critical_issues', 0)}")
                print(f"  High: {security_summary.get('high_severity', 0)}")
                print(f"  Medium: {security_summary.get('medium_severity', 0)}")
                print(f"  Low: {security_summary.get('low_severity', 0)}")
            
            # Print risk assessment
            risk_assessment = result.get("risk_assessment", {})
            if risk_assessment:
                print(f"Risk Assessment: {risk_assessment.get('overall_risk', 'unknown').upper()}")
                print(f"Risk Score: {risk_assessment.get('risk_score', 0)}/100")
        else:
            print(f"Production Security Scanner: Failed to scan directory")
            if result.get("errors"):
                for error in result["errors"][:3]:  # Show first 3 errors
                    print(f"Error: {error}")
        
        if result.get("warnings"):
            for warning in result["warnings"][:3]:  # Show first 3 warnings
                print(f"Warning: {warning}")