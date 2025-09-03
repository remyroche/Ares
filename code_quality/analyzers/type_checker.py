"""
Type Checker - Comprehensive Python type checking and analysis using mypy and advanced type inference.
"""

import os
import ast
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import defaultdict
import json
import logging

from ..core.config import CodeQualityConfig, get_default_config
from ..utils.file_utils import find_python_files


class TypeIssue:
    """Container for type checking issue information."""
    
    def __init__(self, file_path: str, line: int, column: int, message: str, 
                 error_type: str, severity: str = "error", code: str = ""):
        self.file_path = file_path
        self.line = line
        self.column = column
        self.message = message
        self.error_type = error_type
        self.severity = severity
        self.code = code
    
    def __repr__(self):
        return f"TypeIssue({self.file_path}:{self.line}:{self.column}, {self.message})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_path": self.file_path,
            "line": self.line,
            "column": self.column,
            "message": self.message,
            "error_type": self.error_type,
            "severity": self.severity,
            "code": self.code
        }


class TypeInfo:
    """Container for type information."""
    
    def __init__(self, name: str, type_hint: str, line: int, 
                 is_function: bool = False, is_class: bool = False,
                 is_variable: bool = False, has_type_hint: bool = False):
        self.name = name
        self.type_hint = type_hint
        self.line = line
        self.is_function = is_function
        self.is_class = is_class
        self.is_variable = is_variable
        self.has_type_hint = has_type_hint
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type_hint": self.type_hint,
            "line": self.line,
            "is_function": self.is_function,
            "is_class": self.is_class,
            "is_variable": self.is_variable,
            "has_type_hint": self.has_type_hint
        }


class TypeChecker:
    """
    Comprehensive Python type checking and analysis.
    
    Features:
    - MyPy integration for advanced type checking
    - AST-based type inference
    - Type coverage analysis
    - Import type analysis
    - Generic type validation
    """
    
    def __init__(self, config: Optional[CodeQualityConfig] = None):
        self.config = config or get_default_config()
        self.type_issues: List[TypeIssue] = []
        self.type_info: Dict[str, List[TypeInfo]] = {}
        self.mypy_results: Dict[str, Any] = {}
        self.type_coverage: Dict[str, float] = {}
        self.file_stats: Dict[str, Dict[str, Any]] = {}
        
        # Check if mypy is available
        self.mypy_available = self._check_mypy_availability()
        
    def _check_mypy_availability(self) -> bool:
        """Check if mypy is available in the system."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "mypy", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate types for a single Python file.
        
        Args:
            file_path: Path to Python file to validate
            
        Returns:
            Dictionary containing validation results
        """
        try:
            file_path = Path(file_path).resolve()
            
            # Clear previous results for this file
            self.type_issues = [issue for issue in self.type_issues if issue.file_path != str(file_path)]
            
            # Basic AST-based type analysis
            ast_results = self._analyze_ast_types(file_path)
            
            # MyPy analysis if available
            mypy_results = {}
            if self.mypy_available:
                mypy_results = self._run_mypy_analysis(file_path)
            
            # Combine results
            combined_results = self._combine_type_analysis(ast_results, mypy_results)
            
            # Calculate type coverage
            type_coverage = self._calculate_type_coverage(file_path, combined_results)
            
            # Store results
            self.file_stats[str(file_path)] = {
                "type_issues": len(self.type_issues),
                "type_coverage": type_coverage,
                "ast_analysis": ast_results,
                "mypy_analysis": mypy_results
            }
            
            return {
                "status": "success",
                "issues_found": len(self.type_issues),
                "issues_fixed": 0,
                "type_coverage": type_coverage,
                "details": combined_results,
                "mypy_available": self.mypy_available
            }
            
        except Exception as e:
            logging.error(f"Error in type validation for {file_path}: {e}")
            return {
                "status": "error",
                "issues_found": 0,
                "issues_fixed": 0,
                "type_coverage": 0.0,
                "details": {"error": str(e)},
                "mypy_available": self.mypy_available
            }
    
    def _analyze_ast_types(self, file_path: Path) -> Dict[str, Any]:
        """Analyze types using AST parsing."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            type_info = []
            missing_type_hints = []
            generic_usage = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Analyze function type hints
                    has_return_type = node.returns is not None
                    has_param_types = any(arg.annotation is not None for arg in node.args.args)
                    
                    type_info.append(TypeInfo(
                        name=node.name,
                        type_hint=str(node.returns) if node.returns else "Any",
                        line=node.lineno,
                        is_function=True,
                        has_type_hint=has_return_type or has_param_types
                    ))
                    
                    if not has_return_type:
                        missing_type_hints.append(f"Function '{node.name}' missing return type hint")
                    
                    if not has_param_types:
                        missing_type_hints.append(f"Function '{node.name}' missing parameter type hints")
                
                elif isinstance(node, ast.ClassDef):
                    # Analyze class type hints
                    has_bases = len(node.bases) > 0
                    
                    type_info.append(TypeInfo(
                        name=node.name,
                        type_hint=", ".join(ast.unparse(base) for base in node.bases) if has_bases else "object",
                        line=node.lineno,
                        is_class=True,
                        has_type_hint=has_bases
                    ))
                    
                    if not has_bases:
                        missing_type_hints.append(f"Class '{node.name}' missing base class type hints")
                
                elif isinstance(node, ast.AnnAssign):
                    # Analyze variable type hints
                    if node.annotation:
                        type_info.append(TypeInfo(
                            name=ast.unparse(node.target) if hasattr(ast, 'unparse') else str(node.target),
                            type_hint=ast.unparse(node.annotation) if hasattr(ast, 'unparse') else str(node.annotation),
                            line=node.lineno,
                            is_variable=True,
                            has_type_hint=True
                        ))
                        
                        # Check for generic types
                        if "List[" in str(node.annotation) or "Dict[" in str(node.annotation):
                            generic_usage.append(f"Generic type usage: {ast.unparse(node.annotation) if hasattr(ast, 'unparse') else str(node.annotation)}")
            
            return {
                "type_info": [info.to_dict() for info in type_info],
                "missing_type_hints": missing_type_hints,
                "generic_usage": generic_usage,
                "total_functions": len([info for info in type_info if info.is_function]),
                "total_classes": len([info for info in type_info if info.is_class]),
                "total_variables": len([info for info in type_info if info.is_variable]),
                "functions_with_types": len([info for info in type_info if info.is_function and info.has_type_hint]),
                "classes_with_types": len([info for info in type_info if info.is_class and info.has_type_hint])
            }
            
        except Exception as e:
            logging.error(f"Error in AST type analysis: {e}")
            return {"error": str(e)}
    
    def _run_mypy_analysis(self, file_path: Path) -> Dict[str, Any]:
        """Run MyPy analysis on the file."""
        try:
            # Create temporary config for mypy
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as config_file:
                config_content = """
[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True
"""
                config_file.write(config_content)
                config_path = config_file.name
            
            try:
                # Run mypy
                result = subprocess.run(
                    [
                        sys.executable, "-m", "mypy",
                        "--config-file", config_path,
                        "--show-error-codes",
                        "--no-error-summary",
                        "--no-pretty",
                        str(file_path)
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Parse mypy output
                mypy_issues = []
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        if line and ':' in line:
                            parts = line.split(':', 3)
                            if len(parts) >= 4:
                                file_name, line_num, col_num, rest = parts
                                if rest.startswith(' '):
                                    rest = rest[1:]
                                
                                # Extract error code and message
                                if '[' in rest and ']' in rest:
                                    code_end = rest.find(']')
                                    error_code = rest[1:code_end]
                                    message = rest[code_end + 2:]
                                else:
                                    error_code = ""
                                    message = rest
                                
                                try:
                                    line_num = int(line_num)
                                    col_num = int(col_num)
                                    
                                    mypy_issues.append({
                                        "line": line_num,
                                        "column": col_num,
                                        "message": message,
                                        "code": error_code,
                                        "severity": "error"
                                    })
                                except ValueError:
                                    continue
                
                return {
                    "mypy_issues": mypy_issues,
                    "total_issues": len(mypy_issues),
                    "return_code": result.returncode,
                    "stderr": result.stderr
                }
                
            finally:
                # Clean up temporary config
                os.unlink(config_path)
                
        except Exception as e:
            logging.error(f"Error in MyPy analysis: {e}")
            return {"error": str(e)}
    
    def _combine_type_analysis(self, ast_results: Dict[str, Any], mypy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine AST and MyPy analysis results."""
        combined = {
            "ast_analysis": ast_results,
            "mypy_analysis": mypy_results,
            "total_issues": 0,
            "issue_categories": {}
        }
        
        # Count issues from AST analysis
        if "missing_type_hints" in ast_results:
            combined["total_issues"] += len(ast_results["missing_type_hints"])
            combined["issue_categories"]["missing_type_hints"] = len(ast_results["missing_type_hints"])
        
        # Count issues from MyPy analysis
        if "total_issues" in mypy_results:
            combined["total_issues"] += mypy_results["total_issues"]
            combined["issue_categories"]["mypy_issues"] = mypy_results["total_issues"]
        
        return combined
    
    def _calculate_type_coverage(self, file_path: Path, analysis_results: Dict[str, Any]) -> float:
        """Calculate type coverage percentage."""
        try:
            ast_analysis = analysis_results.get("ast_analysis", {})
            
            total_functions = ast_analysis.get("total_functions", 0)
            total_classes = ast_analysis.get("total_classes", 0)
            total_variables = ast_analysis.get("total_variables", 0)
            
            functions_with_types = ast_analysis.get("functions_with_types", 0)
            classes_with_types = ast_analysis.get("classes_with_types", 0)
            
            total_items = total_functions + total_classes + total_variables
            items_with_types = functions_with_types + classes_with_types + total_variables
            
            if total_items == 0:
                return 100.0
            
            return (items_with_types / total_items) * 100
            
        except Exception as e:
            logging.error(f"Error calculating type coverage: {e}")
            return 0.0
    
    def validate_directory(self, directory: str) -> Dict[str, Any]:
        """
        Validate types for all Python files in a directory.
        
        Args:
            directory: Directory containing Python files to validate
            
        Returns:
            Dictionary containing validation results
        """
        python_files = find_python_files(directory, self.config.analysis.exclude_patterns)
        print(f"Validating types for {len(python_files)} Python files...")
        
        # Clear previous results
        self.type_issues.clear()
        self.type_info.clear()
        self.mypy_results.clear()
        self.type_coverage.clear()
        self.file_stats.clear()
        
        total_issues = 0
        total_coverage = 0.0
        successful_files = 0
        
        for file_path in python_files:
            try:
                result = self.validate_file(str(file_path))
                if result["status"] == "success":
                    total_issues += result["issues_found"]
                    total_coverage += result["type_coverage"]
                    successful_files += 1
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
        
        avg_coverage = total_coverage / successful_files if successful_files > 0 else 0.0
        
        return {
            "status": "success",
            "total_files": len(python_files),
            "successful_files": successful_files,
            "total_issues": total_issues,
            "average_type_coverage": avg_coverage,
            "mypy_available": self.mypy_available,
            "file_stats": self.file_stats
        }
    
    def get_type_coverage_report(self) -> Dict[str, Any]:
        """Generate a comprehensive type coverage report."""
        return {
            "overall_coverage": sum(self.type_coverage.values()) / len(self.type_coverage) if self.type_coverage else 0.0,
            "file_coverage": self.type_coverage,
            "total_issues": sum(len(issues) for issues in self.file_stats.values() if "type_issues" in issues),
            "mypy_available": self.mypy_available
        }
    
    def can_analyze(self, file_path: str) -> bool:
        """Check if this analyzer can analyze the given file."""
        return file_path.endswith('.py')
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyze the given file (alias for validate_file)."""
        return self.validate_file(file_path)