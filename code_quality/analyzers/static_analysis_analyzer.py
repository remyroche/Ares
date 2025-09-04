"""
Static Analysis Analyzer - Integrates Pylint, Flake8, MyPy, and Bandit for comprehensive static analysis.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.config import CodeQualityConfig


class StaticAnalysisAnalyzer:
    """
    Comprehensive static analysis using multiple tools:
    - Pylint: Code quality and style analysis
    - Flake8: Style guide enforcement
    - MyPy: Static type checking
    - Bandit: Security vulnerability scanning
    """

    def __init__(self, config: CodeQualityConfig):
        self.config = config
        self.results = {}
        self.tools = {
            "pylint": self._run_pylint,
            "flake8": self._run_flake8,
            "mypy": self._run_mypy,
            "bandit": self._run_bandit,
        }

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single Python file with all static analysis tools."""
        results = {
            "file": file_path,
            "tools": {},
            "summary": {
                "total_issues": 0,
                "critical_issues": 0,
                "warnings": 0,
                "info": 0,
                "security_issues": 0,
            }
        }

        for tool_name, tool_func in self.tools.items():
            try:
                tool_result = tool_func(file_path)
                results["tools"][tool_name] = tool_result
                
                # Aggregate summary
                if tool_result.get("status") == "success":
                    issues = tool_result.get("issues", [])
                    results["summary"]["total_issues"] += len(issues)
                    
                    for issue in issues:
                        severity = issue.get("severity", "info")
                        if severity == "critical":
                            results["summary"]["critical_issues"] += 1
                        elif severity == "warning":
                            results["summary"]["warnings"] += 1
                        elif severity == "info":
                            results["summary"]["info"] += 1
                        
                        if issue.get("category") == "security":
                            results["summary"]["security_issues"] += 1
                            
            except Exception as e:
                results["tools"][tool_name] = {
                    "status": "error",
                    "error": str(e)
                }

        return results

    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """Analyze all Python files in a directory."""
        directory = Path(directory_path)
        python_files = list(directory.rglob("*.py"))
        
        results = {
            "directory": directory_path,
            "files": {},
            "summary": {
                "total_files": len(python_files),
                "total_issues": 0,
                "critical_issues": 0,
                "warnings": 0,
                "info": 0,
                "security_issues": 0,
                "tools_summary": {}
            }
        }

        # Initialize tool summaries
        for tool_name in self.tools.keys():
            results["summary"]["tools_summary"][tool_name] = {
                "files_analyzed": 0,
                "issues_found": 0,
                "errors": 0
            }

        for file_path in python_files:
            file_result = self.analyze_file(str(file_path))
            results["files"][str(file_path)] = file_result
            
            # Update summary
            file_summary = file_result["summary"]
            results["summary"]["total_issues"] += file_summary["total_issues"]
            results["summary"]["critical_issues"] += file_summary["critical_issues"]
            results["summary"]["warnings"] += file_summary["warnings"]
            results["summary"]["info"] += file_summary["info"]
            results["summary"]["security_issues"] += file_summary["security_issues"]
            
            # Update tool summaries
            for tool_name, tool_result in file_result["tools"].items():
                if tool_result.get("status") == "success":
                    results["summary"]["tools_summary"][tool_name]["files_analyzed"] += 1
                    results["summary"]["tools_summary"][tool_name]["issues_found"] += len(tool_result.get("issues", []))
                else:
                    results["summary"]["tools_summary"][tool_name]["errors"] += 1

        return results

    def _run_pylint(self, file_path: str) -> Dict[str, Any]:
        """Run Pylint analysis on a file."""
        try:
            # Create temporary config file for Pylint
            with tempfile.NamedTemporaryFile(mode='w', suffix='.rc', delete=False) as config_file:
                config_file.write("""
[MESSAGES CONTROL]
disable=C0114,C0116,R0903,W0613,C0103

[FORMAT]
max-line-length=120

[DESIGN]
max-args=10
max-locals=20
max-returns=6
max-branches=15
max-statements=60
""")
                config_path = config_file.name

            cmd = [
                sys.executable, "-m", "pylint",
                "--rcfile", config_path,
                "--output-format=json",
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Clean up config file
            Path(config_path).unlink(missing_ok=True)
            
            if result.returncode in [0, 1, 2, 4, 8, 16, 32]:  # Valid pylint exit codes
                try:
                    issues = json.loads(result.stdout) if result.stdout.strip() else []
                except json.JSONDecodeError:
                    issues = []
                
                return {
                    "status": "success",
                    "issues": [
                        {
                            "line": issue.get("line", 0),
                            "column": issue.get("column", 0),
                            "message": issue.get("message", ""),
                            "severity": self._map_pylint_severity(issue.get("type", "info")),
                            "category": "code_quality",
                            "code": issue.get("message-id", ""),
                            "symbol": issue.get("symbol", "")
                        }
                        for issue in issues
                    ],
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                return {
                    "status": "error",
                    "error": f"Pylint failed with return code {result.returncode}",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "Pylint analysis timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _run_flake8(self, file_path: str) -> Dict[str, Any]:
        """Run Flake8 analysis on a file."""
        try:
            cmd = [
                sys.executable, "-m", "flake8",
                "--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s",
                "--max-line-length=120",
                "--extend-ignore=E203,W503",
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            issues = []
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if ':' in line:
                        parts = line.split(':', 3)
                        if len(parts) >= 4:
                            issues.append({
                                "line": int(parts[1]) if parts[1].isdigit() else 0,
                                "column": int(parts[2]) if parts[2].isdigit() else 0,
                                "message": parts[3].strip(),
                                "severity": self._map_flake8_severity(parts[3].split()[0] if parts[3].split() else ""),
                                "category": "style",
                                "code": parts[3].split()[0] if parts[3].split() else ""
                            })
            
            return {
                "status": "success",
                "issues": issues,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "Flake8 analysis timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _run_mypy(self, file_path: str) -> Dict[str, Any]:
        """Run MyPy type checking on a file."""
        try:
            cmd = [
                sys.executable, "-m", "mypy",
                "--show-error-codes",
                "--no-error-summary",
                "--ignore-missing-imports",
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            issues = []
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if ':' in line and 'error:' in line:
                        parts = line.split(':', 3)
                        if len(parts) >= 4:
                            issues.append({
                                "line": int(parts[1]) if parts[1].isdigit() else 0,
                                "column": 0,  # MyPy doesn't provide column info
                                "message": parts[3].strip(),
                                "severity": "warning",
                                "category": "type_checking",
                                "code": "mypy"
                            })
            
            return {
                "status": "success",
                "issues": issues,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "MyPy analysis timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _run_bandit(self, file_path: str) -> Dict[str, Any]:
        """Run Bandit security analysis on a file."""
        try:
            cmd = [
                sys.executable, "-m", "bandit",
                "-f", "json",
                "-r", file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            issues = []
            if result.stdout:
                try:
                    bandit_output = json.loads(result.stdout)
                    for issue in bandit_output.get("results", []):
                        issues.append({
                            "line": issue.get("line_number", 0),
                            "column": 0,
                            "message": issue.get("issue_text", ""),
                            "severity": self._map_bandit_severity(issue.get("issue_severity", "LOW")),
                            "category": "security",
                            "code": issue.get("test_id", ""),
                            "confidence": issue.get("issue_confidence", "LOW")
                        })
                except json.JSONDecodeError:
                    pass
            
            return {
                "status": "success",
                "issues": issues,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "Bandit analysis timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _map_pylint_severity(self, pylint_type: str) -> str:
        """Map Pylint message types to severity levels."""
        severity_map = {
            "error": "critical",
            "warning": "warning",
            "convention": "info",
            "refactor": "info",
            "info": "info"
        }
        return severity_map.get(pylint_type, "info")

    def _map_flake8_severity(self, code: str) -> str:
        """Map Flake8 error codes to severity levels."""
        if code.startswith('E'):
            return "critical"
        elif code.startswith('W'):
            return "warning"
        elif code.startswith('F'):
            return "critical"
        else:
            return "info"

    def _map_bandit_severity(self, severity: str) -> str:
        """Map Bandit severity levels."""
        severity_map = {
            "HIGH": "critical",
            "MEDIUM": "warning",
            "LOW": "info"
        }
        return severity_map.get(severity, "info")

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive static analysis report."""
        return {
            "analyzer": "StaticAnalysisAnalyzer",
            "tools_used": list(self.tools.keys()),
            "results": self.results,
            "summary": self._generate_summary()
        }

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {"status": "no_analysis_performed"}
        
        total_files = 0
        total_issues = 0
        critical_issues = 0
        security_issues = 0
        
        for file_result in self.results.get("files", {}).values():
            total_files += 1
            summary = file_result.get("summary", {})
            total_issues += summary.get("total_issues", 0)
            critical_issues += summary.get("critical_issues", 0)
            security_issues += summary.get("security_issues", 0)
        
        return {
            "total_files_analyzed": total_files,
            "total_issues_found": total_issues,
            "critical_issues": critical_issues,
            "security_issues": security_issues,
            "average_issues_per_file": total_issues / total_files if total_files > 0 else 0
        }